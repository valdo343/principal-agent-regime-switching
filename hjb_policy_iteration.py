"""
hjb_policy_iteration.py

Funciones para resolver y simular el problema del agente.

Este módulo recibe una función de ponderación contratual c(t) y la usa para:
    1. resolver la HJB del agente mediante iteración de políticas;
    2. simular trayectorias bajo la política óptima obtenida;
    3. graficar diagnósticos del agente.
"""

from __future__ import annotations

from typing import Callable, Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from matplotlib.lines import Line2D


plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.titlesize": 20,
})


GridTuple = Tuple[np.ndarray, np.ndarray, np.ndarray]
Params = Dict[str, object]
Contract = Callable[[np.ndarray], np.ndarray]


# ============================================================
# 1. Índices y tamaño de malla
# ============================================================

def get_index(i, j, k, l, N_I, N_P, N_Q):
    """
    Mapea (i,j,k,l) -> r.
    i: índice de impacto I
    j: índice de precio P
    k: índice de inventario restante Q
    l: régimen 0/1
    """
    M = (N_I + 1) * (N_P + 1) * (N_Q + 1)
    return i + j * (N_I + 1) + k * (N_I + 1) * (N_P + 1) + l * M


def reverse_index(r, N_I, N_P, N_Q):
    """
    Mapea r -> (i,j,k,l).
    """
    M = (N_I + 1) * (N_P + 1) * (N_Q + 1)
    l = r // M
    rem = r % M

    k = rem // ((N_I + 1) * (N_P + 1))
    rem2 = rem % ((N_I + 1) * (N_P + 1))
    j = rem2 // (N_I + 1)
    i = rem2 % (N_I + 1)

    return i, j, k, l


def grid_sizes(grids):
    I_grid, P_grid, Q_grid = grids
    N_I, N_P, N_Q = len(I_grid) - 1, len(P_grid) - 1, len(Q_grid) - 1
    M = (N_I + 1) * (N_P + 1) * (N_Q + 1)
    Total_Size = 2 * M
    return N_I, N_P, N_Q, M, Total_Size


# ============================================================
# 2. Renglones discretos de A(q) y B(q)
# ============================================================

def build_A_row_entries(i, j, k, l, q, grids, params, c_val):
    """
    Construye SOLO el renglón r de A(q) como lista de pares (columna, valor).

    Mantiene la estructura original:
        A[r,r] = 1 + dtau*(Sigma + lambda_l + P gamma c)
        vecinos espaciales negativos
        acoplamiento de régimen negativo

    No usa log(theta).
    """
    I_grid, P_grid, Q_grid = grids
    N_I, N_P, N_Q, M, Total_Size = grid_sizes(grids)

    dI = I_grid[1] - I_grid[0]
    dP = P_grid[1] - P_grid[0]
    dQ = Q_grid[1] - Q_grid[0]

    dtau = params["dtau"]
    sigma = params["sigma"]
    rho_l = params["rho"][l]
    kappa_l = params["kappa"][l]
    lam_l = params["lambda"][l]
    gamma = params["gamma"]

    r = get_index(i, j, k, l, N_I, N_P, N_Q) # Índice mapeado del nodo (i,j,k,l)
    h = 1 - l # Régimen opuesto

    I_val = I_grid[i] # Valor de I_i
    P_val = P_grid[j] # Valor de P_j

    # Coeficientes de derivadas parciales.
    v_I = -rho_l * I_val + kappa_l * q
    v_P = v_I
    v_Q = -q

    # Valores para la discretización de coeficientes positivos
    alpha_I = max(-v_I, 0.0) / dI
    beta_I  = max( v_I, 0.0) / dI

    alpha_P = (sigma**2) / (2.0 * dP**2) + max(-v_P, 0.0) / dP
    beta_P  = (sigma**2) / (2.0 * dP**2) + max( v_P, 0.0) / dP

    alpha_Q = max(-v_Q, 0.0) / dQ
    beta_Q  = max( v_Q, 0.0) / dQ

    # Fronteras: se apaga el flujo que sale del dominio.
    if i == 0:
        alpha_I = 0.0
    if i == N_I:
        beta_I = 0.0

    if j == 0:
        alpha_P = 0.0
    if j == N_P:
        beta_P = 0.0

    if k == 0:
        alpha_Q = 0.0
    if k == N_Q:
        beta_Q = 0.0

    Sigma = alpha_I + beta_I + alpha_P + beta_P + alpha_Q + beta_Q

    entries = []

    # Entrada diagonal
    entries.append((r, 1.0 + dtau * (Sigma + lam_l + P_val*gamma*c_val)))

    # Entradas de vecinos en I
    if i > 0:
        entries.append((get_index(i - 1, j, k, l, N_I, N_P, N_Q), -dtau * alpha_I))
    if i < N_I:
        entries.append((get_index(i + 1, j, k, l, N_I, N_P, N_Q), -dtau * beta_I))

    # Entradas de vecinos en P
    if j > 0:
        entries.append((get_index(i, j - 1, k, l, N_I, N_P, N_Q), -dtau * alpha_P))
    if j < N_P:
        entries.append((get_index(i, j + 1, k, l, N_I, N_P, N_Q), -dtau * beta_P))

    # Entradas de vecinos en Q
    if k > 0:
        entries.append((get_index(i, j, k - 1, l, N_I, N_P, N_Q), -dtau * alpha_Q))
    if k < N_Q:
        entries.append((get_index(i, j, k + 1, l, N_I, N_P, N_Q), -dtau * beta_Q))

    # Entrada asociada al cambio de régimen
    entries.append((get_index(i, j, k, h, N_I, N_P, N_Q), -dtau * lam_l))

    return entries


def compute_B_value(j, q, grids, params):
    """
    B(q,c) diagonal.

    Se hace
        b[r] = [1 + dtau * gamma * (Pq + eta/2 q^2)] * Theta_n[r]
    """
    _, P_grid, _ = grids
    P_val = P_grid[j]

    dtau = params["dtau"]
    gamma = params["gamma"]
    eta = params["eta"]

    return 1.0 + dtau * gamma * (P_val*q + 0.5 * eta * q**2)


def row_dot(entries, x):
    """
    Producto de un renglón disperso con un vector x.
    """
    return sum(value * x[col] for col, value in entries)


# ============================================================
# 3. Política discreta
# ============================================================

def compute_policy_by_discrete_argmin(Theta_guess, Theta_n, q_guess, grids, params, c_val):
    """
    Calcula q* nodo a nodo resolviendo el problema discreto:

        q* in argmin_q { -[A(q) Theta_guess]_r + B_r(q) Theta_n[r] }
    """
    I_grid, P_grid, Q_grid = grids
    N_I, N_P, N_Q, M, Total_Size = grid_sizes(grids)

    q_max = params.get("q_max", 10.0)
    n_q_candidates = params.get("n_q_candidates", 50)
    tie_tol = params.get("tie_tol", 1e-12)
    

    if q_guess is None:
        q_guess = np.zeros(Total_Size)

    q_policy = np.zeros(Total_Size)

    for l in range(2):
        for i in range(N_I + 1):
            for j in range(N_P + 1):
                for k in range(N_Q + 1):
                    r = get_index(i, j, k, l, N_I, N_P, N_Q)
                    Q_val = Q_grid[k]

                    # Si no queda inventario, no ejecutamos.
                    if Q_val <= 1e-14:
                        q_policy[r] = 0.0
                        continue

                    q_max_node = q_max

                    q_candidates = np.linspace(0.0, q_max_node, n_q_candidates)

                    F_values = np.empty_like(q_candidates) # Vector de valores de la función objetivo para cada candidato q

                    for m, q in enumerate(q_candidates):
                        entries = build_A_row_entries(i, j, k, l, q, grids, params, c_val) # Construye el renglón r de A(q) para el candidato q en q_candidates
                        Atheta = row_dot(entries, Theta_guess) # Calcula A(q)Theta_guess para el renglón r
                        Bval = compute_B_value(j, q, grids, params) # Calcula B(q) para el renglón r

                        # Objetivo discreto:
                        # inf_q { -A(q)Theta^{n+1} + B(q)Theta^n }
                        F_values[m] = -Atheta + Bval * Theta_n[r] # Valor de la función objetivo para el candidato q

                    best = np.argmin(F_values) # Índice del candidato q que minimiza la función objetivo

                    q_policy[r] = q_candidates[best]

    return q_policy


# ============================================================
# 4. Sistema lineal y residuales
# ============================================================

def build_A_and_B(q_policy, grids, params, c_val):
    """
    Construye A(q) y el vector diagonal B_vec(q,c).
    El lado derecho es:
        rhs = B_vec * Theta_n
    """
    I_grid, P_grid, Q_grid = grids
    N_I, N_P, N_Q, M, Total_Size = grid_sizes(grids)

    A = sp.lil_matrix((Total_Size, Total_Size))
    B_vec = np.zeros(Total_Size)

    for l in range(2):
        for i in range(N_I + 1):
            for j in range(N_P + 1):
                for k in range(N_Q + 1):
                    r = get_index(i, j, k, l, N_I, N_P, N_Q)
                    q = q_policy[r]

                    entries = build_A_row_entries(i, j, k, l, q, grids, params, c_val)
                    for col, value in entries:
                        A[r, col] += value # Llenamos la matriz A con los valores del renglón r correspondiente al nodo (i,j,k,l)

                    B_vec[r] = compute_B_value(j, q, grids, params)

    return A.tocsr(), B_vec 


def discrete_residual(A, B_vec, Theta_np1, Theta_n):
    """
    Residual discreto:
        R = A @ Theta_np1 - B_vec * Theta_n
    """
    R = A @ Theta_np1 - B_vec * Theta_n
    return np.max(np.abs(R)), R


def m_matrix_diagnostics(A):
    """
    Diagnóstico de M-matriz.
    Revisa:
      - diagonal positiva,
      - entradas no-diagonales no positivas,
      - dominancia diagonal por filas.
    """
    A_csr = A.tocsr()
    diag = A_csr.diagonal() # Diagonal de la matriz A

    A_coo = A_csr.tocoo()
    off_mask = A_coo.row != A_coo.col # Máscara booleana para seleccionar las entradas fuera de la diagonal
    off_data = A_coo.data[off_mask] # Datos de las entradas fuera de la diagonal

    max_offdiag = np.max(off_data) if off_data.size > 0 else 0.0 # Valor máxiomo de las entradas fuera de la diagonal

    abs_A = abs(A_csr) # Matriz con valores absolutos de A
    off_abs_sum = abs_A.sum(axis=1).A.ravel() - np.abs(diag) # Suma de los valores absolutos de las entradas fuera de la diagonal por fila
    dd_margin = diag - off_abs_sum # Margen de dominancia diagonal por fila

    return {
        "min_diag": float(np.min(diag)),
        "max_offdiag": float(max_offdiag),
        "min_diag_dominance_margin": float(np.min(dd_margin)),
        "diag_positive": bool(np.all(diag > 0.0)), # Verifica si todas las entradas de la diagonal son positivas
        "offdiag_nonpositive": bool(max_offdiag <= 1e-14), # Verifica si todas las entradas fuera de la diagonal son no positivas
        "diag_dominant": bool(np.all(dd_margin >= -1e-12)), # Verifica si todas las filas cumplen la condición de dominancia diagonal
    }


# ============================================================
# 5. Iteración de políticas
# ============================================================

def pol_iter_step(Theta_n, q_guess, grids, params, c_val, step=None):
    """
    Resuelve un paso de tiempo:

        inf_q {-A(q)Theta^{n+1} + B(q)Theta^n} = 0.

    Para cada iteración:
      1. Calcula q_new por búsqueda discreta usando Theta_guess.
      2. Construye A(q_new), B(q_new).
      3. Resuelve A @ Theta_new = B @ Theta_n.
      4. Calcula residual discreto R = A @ Theta_new - B @ Theta_n.
      5. Repite hasta que Theta y q se estabilicen.
    """
    _, _, _, _, Total_Size = grid_sizes(grids)

    tol = params.get("tol", 1e-6) # Tolerancia de convergencia
    max_iter = params.get("max_iter", 30)

    q_max = params.get("q_max", 10.0)
    n_q_candidates = params.get("n_q_candidates", 50)
    q_step = q_max / max(n_q_candidates - 1, 1) # Tamaño del paso entre candidatos de q
    q_tol = params.get("q_tol", 0.5 * q_step + 1e-14) # Tolerancia para la convergencia de q. Nos detenemos cuando el cambio en q es menor que medio paso entre candidatos de q.

    Theta_guess = Theta_n.copy() # Inicializamos la primera aproximación de Theta_{n+1} como Theta_n
    q_old = np.zeros(Total_Size) if q_guess is None else q_guess.copy() # Inicializamos la primera aproximación de q como un vector de ceros o como q_guess si se proporciona

    history = []

    for it in range(max_iter):
        # 1. Política óptima discreta para el valor actual
        q_new = compute_policy_by_discrete_argmin(
            Theta_guess=Theta_guess,
            Theta_n=Theta_n,
            q_guess=q_old,
            grids=grids,
            params=params,
            c_val=c_val,
        )

        # 2. Sistema lineal bajo política fija
        A, B_vec = build_A_and_B(q_new, grids, params, c_val)
        rhs = B_vec * Theta_n

        # Diagnóstico M-matriz
        if params.get("print_m_matrix_warnings", True):
            diag_info = m_matrix_diagnostics(A)
            if not (diag_info["diag_positive"] and diag_info["offdiag_nonpositive"] and diag_info["diag_dominant"]):
                print("   !!! Advertencia: diagnóstico M-matriz falló:", diag_info)

        # 3. Resolver sistema lineal para la siguiente aproximación de Theta_{n+1}
        Theta_new = spla.spsolve(A, rhs)

        # 4. Residual discreto con la política usada para resolver
        fixed_policy_residual, _ = discrete_residual(A, B_vec, Theta_new, Theta_n)

        # 5. Errores de actualización
        theta_rel_error = np.max(
            np.abs(Theta_new - Theta_guess) / np.maximum(np.abs(Theta_new), 1e-12)
        )
        q_abs_error = np.max(np.abs(q_new - q_old))

        history.append({
            "iteration": it,
            "theta_rel_error": float(theta_rel_error),
            "q_abs_error": float(q_abs_error),
            "fixed_policy_residual": float(fixed_policy_residual),
        })

        print(
            f"   Iter {it:02d}: "
            f"err_theta={theta_rel_error:.3e}, "
            f"err_q={q_abs_error:.3e}, "
            f"res_fijo={fixed_policy_residual:.3e} "
        )

        # Nos detenemos cuando tanto Theta como q se estabilizan.
        if theta_rel_error < tol and q_abs_error <= q_tol:
            print(f"   >>> Convergencia en iteración {it + 1}")
            return Theta_new, q_new, history

        Theta_guess = Theta_new
        q_old = q_new

    print(f"   !!! Máximo de iteraciones alcanzado. Último err_theta={theta_rel_error:.3e}, err_q={q_abs_error:.3e}")
    return Theta_guess, q_old, history


# ============================================================
# 6. Condición terminal y solver HJB
# ============================================================

def terminal_condition(grids, params):
    """
    Condición terminal:
        theta(0,I,P,Q,l) = exp(gamma*(P*Q + A_pen*Q^2))

    Aquí tau=0 corresponde a t=T.
    """
    I_grid, P_grid, Q_grid = grids
    N_I, N_P, N_Q, M, Total_Size = grid_sizes(grids)

    gamma = params["gamma"]
    A_pen = params["A"]

    Theta_0 = np.zeros(Total_Size)

    for r in range(Total_Size):
        i, j, k, l = reverse_index(r, N_I, N_P, N_Q)
        P_val = P_grid[j]
        Q_val = Q_grid[k]

        exponent = gamma * (P_val * Q_val + A_pen * Q_val**2)
        exponent = min(exponent, 500.0)
        Theta_0[r] = np.exp(exponent)

    return Theta_0


def solve_hjb(grids, params, contract_func):
    """
    Resuelve la HJB en tau.

    Implementa:
        tau_grid = np.arange(n_steps) * dt
        t_grid = T - tau_grid
        c_t_array = contract_func(t_grid)
    """
    dt = params["dtau"]
    T = params["T"]
    n_steps = int(round(T / dt))

    tau_grid = np.arange(n_steps) * dt
    t_grid = T - tau_grid
    c_t_array = contract_func(t_grid) # Evaluamos la función de ponderación contratual c(t) en los tiempos t_grid

    _, _, _, _, Total_Size = grid_sizes(grids)

    all_thetas = np.zeros((n_steps + 1, Total_Size)) # Almacenará Theta para cada paso de tau, incluyendo la condición terminal en tau=0
    all_qs = np.zeros((n_steps, Total_Size)) # Almacenará q* para cada paso de tau, no incluye la condición terminal ya que no hay control en tau=0.
    step_histories = []

    all_thetas[0, :] = terminal_condition(grids, params) # Inicializamos la condición terminal en tau=0 (t=T)

    q_current = np.zeros(Total_Size)

    print(f"Iniciando resolución: n_steps={n_steps}, dtau={dt}, q_max={params.get('q_max', 10.0)}")
    print("Ojo: all_thetas[0] es tau=0, es decir, condición terminal en t=T.")

    for step in range(n_steps):
        print(f"\n--- Paso tau {step + 1}/{n_steps}; tau={tau_grid[step]:.4f}; t={t_grid[step]:.4f}; c={c_t_array[step]:.4f} ---")

        Theta_new, q_new, hist = pol_iter_step(
            Theta_n=all_thetas[step, :],
            q_guess=q_current,
            grids=grids,
            params=params,
            c_val=float(c_t_array[step]),
            step=step,
        )

        all_thetas[step + 1, :] = Theta_new
        all_qs[step, :] = q_new
        q_current = q_new
        step_histories.append(hist)

    return {
        "all_thetas": all_thetas,
        "all_qs": all_qs,
        "tau_grid": tau_grid,
        "t_grid": t_grid,
        "c_t_array": c_t_array,
        "histories": step_histories,
    }


# ============================================================
# 7. Simulación hacia adelante
# ============================================================

def nearest_index(grid, x):
    """
    Encuentra el índice del valor más cercano a x en la malla grid.
    """
    return int(np.argmin(np.abs(grid - x)))


def get_policy_nearest(q_vec, state, regime, grids):
    """
    Evalúa q por vecino más cercano en la malla.
    state = (I, P, Q)
    """
    I, P, Q = state
    I_grid, P_grid, Q_grid = grids
    N_I, N_P, N_Q, _, _ = grid_sizes(grids)

    i = nearest_index(I_grid, np.clip(I, I_grid[0], I_grid[-1])) # Vecino más cercano en la malla de I
    j = nearest_index(P_grid, np.clip(P, P_grid[0], P_grid[-1])) # Vecino más cercano en la malla de P
    k = nearest_index(Q_grid, np.clip(Q, Q_grid[0], Q_grid[-1])) # Vecino más cercano en la malla de Q

    r = get_index(i, j, k, regime, N_I, N_P, N_Q) # índice de vecino más cercano
    return q_vec[r]


def simulate_one_path(result, grids, params, contract_func, seed=123, I0=0.0, P0=4.0, Q0=1.0, regime0=0):
    """
    Simula una trayectoria hacia adelante en tiempo real t.

    Como all_qs está ordenado en tau, para tiempo real t_n usamos:
        policy_idx = n_steps - 1 - n.
    """
    rng = np.random.default_rng(seed)

    all_qs = result["all_qs"]
    dt = params["dtau"]
    T = params["T"]
    n_steps = all_qs.shape[0]

    I = float(I0)
    P = float(P0)
    Q = float(Q0)
    l = int(regime0)
    X = 0.0

    I_path = [I]
    P_path = [P]
    Q_path = [Q]
    regime_path = [l]
    X_path = [X]
    q_path = []

    clip_count = 0

    for n in range(n_steps):
        t = n * dt # Paso n-ésimo

        policy_idx = n_steps - 1 - n
        q_policy = get_policy_nearest(all_qs[policy_idx], (I, P, Q), l, grids) # Política óptima q* en el paso n-ésimo evaluada en el estado actual (I,P,Q,l).

        # En simulación física no ejecutamos más de lo que queda.
        q_exec = min(q_policy, Q / dt) if Q > 0.0 else 0.0

        c_val = float(contract_func(np.array([t]))[0]) # Valor del contrato al tiempo t

        dW = np.sqrt(dt) * rng.normal() # Incremento del movimiento browniano

        rho_l = params["rho"][l]
        kappa_l = params["kappa"][l]
        lam_l = params["lambda"][l]

        dI = (-rho_l * I + kappa_l * q_exec) * dt # Incremento en I
        dP = params["sigma"] * dW + dI # Incremento en P
        dQ = -q_exec * dt # Incremento en Q
        dX = P * (c_val - q_exec) * dt - 0.5 * params["eta"] * q_exec**2 * dt # Incremento en X

        I = I + dI # Actualizamos el impacto I
        P = P + dP # Actualizamos el precio P
        Q = max(Q + dQ, 0.0) # Actualizamos el inventario Q, asegurando que no sea negativo
        X = X + dX # Actualizamos la riqueza acumulada X

        # Cambio de régimen con probabilidad lam_l * dt
        if rng.uniform() < lam_l * dt:
            l = 1 - l

        # Contar salidas del dominio de la malla para evaluar errores de interpolación
        I_grid, P_grid, Q_grid = grids
        if I < I_grid[0] or I > I_grid[-1] or P < P_grid[0] or P > P_grid[-1] or Q < Q_grid[0] or Q > Q_grid[-1]:
            clip_count += 1

        q_path.append(q_exec)
        I_path.append(I)
        P_path.append(P)
        Q_path.append(Q)
        regime_path.append(l)
        X_path.append(X)

    return {
        "t": np.linspace(0.0, T, n_steps + 1),
        "q": np.array(q_path),
        "I": np.array(I_path),
        "P": np.array(P_path),
        "Q": np.array(Q_path),
        "regime": np.array(regime_path),
        "X": np.array(X_path),
        "clip_count": clip_count,
    }


# ============================================================
# 8. Gráficas de diagnóstico del agente
# ============================================================

def plot_simulated_paths(paths, title="Simulación de Trayectorias de Liquidación Óptima"):
    """
    paths: lista de salidas de simulate_one_path.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(title, fontsize=16)

    for path in paths:
        t = path["t"]
        tq = t[:-1]

        axes[0, 0].plot(tq, path["q"])
        axes[0, 1].plot(t, path["Q"])
        axes[0, 2].plot(t, path["I"])

        axes[1, 0].step(t, path["regime"], where="post")
        axes[1, 1].plot(t, path["P"])
        axes[1, 2].plot(t, path["X"])

    axes[0, 0].set_title(r"Control óptimo $q^*$")
    axes[0, 1].set_title(r"Inventario restante $Q_t$")
    axes[0, 2].set_title(r"Impacto $I_t$")

    axes[1, 0].set_title("Régimen (0=A, 1=B)")
    axes[1, 1].set_title(r"Precio $P_t$")
    axes[1, 2].set_title("Riqueza acumulada")

    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_policy_slice(result, grids, tau_index=-1, regime=0, I_fixed=0.0):
    """
    Heatmap de q*(P,Q) para un tau y régimen dados, con I fijo.
    """
    all_qs = result["all_qs"]
    I_grid, P_grid, Q_grid = grids
    N_I, N_P, N_Q, _, _ = grid_sizes(grids)

    i0 = nearest_index(I_grid, I_fixed)

    q_slice = np.zeros((N_P + 1, N_Q + 1))
    for j in range(N_P + 1):
        for k in range(N_Q + 1):
            r = get_index(i0, j, k, regime, N_I, N_P, N_Q)
            q_slice[j, k] = all_qs[tau_index, r]

    plt.figure(figsize=(8, 5))
    plt.contourf(P_grid, Q_grid, q_slice.T, levels=20)
    plt.colorbar(label=r"$q^*$")
    plt.xlabel(r"$P$")
    plt.ylabel(r"$Q$")
    plt.title(fr"Política $q^*$, tau_index={tau_index}, régimen={regime}, I≈{I_grid[i0]:.2f}")
    plt.grid(alpha=0.2)
    plt.show()


def plot_single_q_colored_by_regime(
    path,
    title=r"Estrategia óptima $q^*$",
    regime_labels=("A", "B"),
    regime_colors=("tab:blue", "tab:orange"),
    figsize=(9, 4.8),
    linewidth=2.2,
    alpha=0.95,
    show_vertical_jumps=True,
    jump_color_mode="next",  # "next", "prev" o "black"
):
    """
    Grafica una sola trayectoria simulada de q^* como función escalonada,
    coloreando cada tramo según el régimen vigente en ese intervalo.

    path: diccionario devuelto por simulate_one_path.
          Se espera:
              path["t"]      -> tiempos, longitud N+1
              path["q"]      -> control, longitud N
              path["regime"] -> régimen, longitud N+1 o N
    """

    t = np.asarray(path["t"])
    q = np.asarray(path["q"])
    regime = np.asarray(path["regime"])

    if len(t) != len(q) + 1:
        raise ValueError("Se espera que len(t) = len(q) + 1.")

    # Si regime tiene longitud N+1, usamos el régimen al inicio de cada intervalo
    if len(regime) == len(q) + 1:
        regime_interval = regime[:-1]
    elif len(regime) == len(q):
        regime_interval = regime
    else:
        raise ValueError("La longitud de regime debe ser N o N+1, donde len(q)=N.")

    fig, ax = plt.subplots(figsize=figsize)

    # Tramos horizontales
    for n in range(len(q)):
        reg = int(regime_interval[n])
        color = regime_colors[reg]

        ax.hlines(
            y=q[n],
            xmin=t[n],
            xmax=t[n + 1],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )

        # Saltos verticales para conectar escalones
        if show_vertical_jumps and n < len(q) - 1:
            if jump_color_mode == "prev":
                jump_color = regime_colors[int(regime_interval[n])]
            elif jump_color_mode == "next":
                jump_color = regime_colors[int(regime_interval[n + 1])]
            else:
                jump_color = "black"

            ax.vlines(
                x=t[n + 1],
                ymin=min(q[n], q[n + 1]),
                ymax=max(q[n], q[n + 1]),
                color=jump_color,
                linewidth=0.9 * linewidth,
                alpha=0.8 * alpha,
            )

    legend_handles = [
        Line2D(
            [0], [0],
            color=regime_colors[i],
            lw=linewidth,
            label=f"Régimen {regime_labels[i]}"
        )
        for i in range(len(regime_labels))
    ]

    ax.legend(handles=legend_handles, title="Régimen", loc="best")
    ax.set_title(title)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$q_t^*$")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_q_fixed_state_by_regime(result, grids, I_fixed=0.0, P_fixed=80.0, Q_fixed=1.0):
    I_grid, P_grid, Q_grid = grids
    N_I, N_P, N_Q, _, _ = grid_sizes(grids)

    i0 = nearest_index(I_grid, I_fixed)
    j0 = nearest_index(P_grid, P_fixed)
    k0 = nearest_index(Q_grid, Q_fixed)

    plt.figure(figsize=(8, 4))

    for regime in [0, 1]:
        r = get_index(i0, j0, k0, regime, N_I, N_P, N_Q)

        t_real = result["t_grid"][::-1]
        q_real = result["all_qs"][::-1, r]

        plt.plot(
            t_real,
            q_real,
            marker="o",
            linewidth=2.4,
            label=f"Régimen {regime}"
        )

    plt.xlabel(r"$t$")
    plt.ylabel(r"$q^*(t,I,P,Q,\alpha)$")
    plt.title(
        fr"Política por régimen: $I\approx{I_grid[i0]:.2f}$, "
        fr"$P\approx{P_grid[j0]:.2f}$, $Q\approx{Q_grid[k0]:.2f}$"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


