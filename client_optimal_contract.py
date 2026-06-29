"""
cliente_contrato_optimo_sencillo.py

Funciones para resolver el problema del cliente.

La lógica es:
    beta -> contrato c_beta(t)
         -> resolver HJB del agente con hjb_policy_iteration.solve_hjb
         -> simular trayectorias con hjb_policy_iteration.simulate_one_path
         -> calcular C0*(c_beta) y J(c_beta)
         -> escoger el beta con menor J.

El problema del cliente se aproxima mediante una búsqueda exhaustiva
sobre una malla finita de coeficientes beta. Por tanto, el contrato
encontrado es óptimo dentro de la malla considerada, no necesariamente
en toda la clase continua de contratos admisibles.

Este archivo debe estar en la misma carpeta que hjb_policy_iteration.py.
"""

from __future__ import annotations

from contextlib import redirect_stdout, nullcontext
from itertools import product
from math import comb
import io
import time

import numpy as np
import pandas as pd
from scipy.special import logsumexp

import hjb_policy_iteration as hjb


# ============================================================
# 1. Utilidades pequeñas
# ============================================================

def logmeanexp(x):
    """
    Calcula log(mean(exp(x))) de forma numéricamente estable.

    En lugar de calcular exp(x) directamente, usamos logsumexp para evitar
    problemas cuando x toma valores grandes.
    """
    x = np.asarray(x, dtype=float)
    return float(logsumexp(x) - np.log(x.size))


def silenciar_si(se_debe_silenciar):
    """
    Regresa un contexto para silenciar o no los prints del solver HJB.

    Uso:
        with silenciar_si(True):
            resultado = hjb.solve_hjb(...)
    """
    if se_debe_silenciar:
        return redirect_stdout(io.StringIO())
    return nullcontext()


# ============================================================
# 2. Familia de contratos Bernstein
# ============================================================

def contrato_bernstein(beta, T=1.0):
    """
    Dado un vector beta de coeficientes no negativos, regresa la función c_beta(t)
    definida en [0,T] por la base de Bernstein.
    Por default se toma T=1
    """
    beta = np.asarray(beta, dtype=float)

    if np.any(beta < 0):
        raise ValueError("Todos los coeficientes beta deben ser no negativos.")

    grado = len(beta) - 1

    def c(t):
        t = np.asarray(t, dtype=float)
        t_clip = np.clip(t, 0.0, T) # Por seguridad numérica, truncamos t al intervalo [0,T].
        salida = np.zeros_like(t_clip, dtype=float)

        for r, beta_r in enumerate(beta):
            salida += (
                beta_r
                * comb(grado, r)
                * (t_clip ** r)
                * ((T - t_clip) ** (grado - r))
            )

        return salida

    return c


def normalizar_beta_integral(beta, T=1.0, integral_objetivo=1.0):
    """
    Reescala beta para que integral_0^T c_beta(t) dt = integral_objetivo.

    Nota:
    Esta fórmula corresponde a la parametrización usada en contrato_bernstein,

        c_beta(t) = sum_r beta_r * C(d,r) * t^r * (T-t)^{d-r},

    cuya integral en [0,T] es

        T^{d+1}/(d+1) * sum_r beta_r.
    """
    beta = np.asarray(beta, dtype=float)
    suma = float(np.sum(beta))

    if suma <= 0.0:
        raise ValueError("No se puede normalizar beta si sum(beta)=0.")

    grado = len(beta) - 1
    integral_actual = (T ** (grado + 1)) * suma / (grado + 1)
    return beta * (integral_objetivo / integral_actual)


def malla_beta_simplex(grado=2, niveles=3, T=1.0, integral_objetivo=1.0):
    """
    Genera candidatos beta no negativos con integral fija.

    Esta opción compara formas contractuales manteniendo fija la escala total.
    Por ejemplo, si integral_objetivo=1, entonces todos los contratos cumplen

        integral_0^T c_beta(t) dt = 1.
    """
    if grado < 1:
        raise ValueError("grado debe ser al menos 1.")
    if niveles < 2:
        raise ValueError("niveles debe ser al menos 2.")

    niveles_grid = np.linspace(0.0, 1.0, niveles)
    candidatos = []
    vistos = set()

    # Revisamos todas las tuplas en niveles_grid de tamaño grado + 1
    for w in product(niveles_grid, repeat=grado + 1):
        w = np.asarray(w, dtype=float)
        suma = float(np.sum(w))

        if suma <= 0.0:
            continue

        # Normalizamos el vector w, es decir, w_0 + ... + w_d = 1.
        w = w / suma

        # Queremos integral = integral_objetivo. Entonces reescalamos w.
        beta = w * ((grado + 1) * integral_objetivo / (T ** (grado + 1)))

        llave = tuple(np.round(beta, 10))
        if llave not in vistos:
            vistos.add(llave)
            candidatos.append(beta)

    return candidatos


def malla_beta_caja(beta_min, beta_max, niveles):
    """
    Genera candidatos beta en una caja distinta para cada coeficiente.

    Ejemplo:
        beta_candidates = malla_beta_caja(
            beta_min=[0.0, 0.0, 0.0],
            beta_max=[2.0, 2.0, 0.5],
            niveles=[7, 7, 4],
        )

    Esto genera:
        beta_0 en [0.0, 2.0] con 7 puntos
        beta_1 en [0.0, 2.0] con 7 puntos
        beta_2 en [0.0, 0.5] con 4 puntos
    
    El número total de contratos evaluados es el producto de las entradas de niveles. Por ejemplo, niveles=[7,7,4] genera 7*7*4 = 196 contratos.
    """
    beta_min = np.asarray(beta_min, dtype=float)
    beta_max = np.asarray(beta_max, dtype=float)
    niveles = np.asarray(niveles, dtype=int)

    if not (len(beta_min) == len(beta_max) == len(niveles)):
        raise ValueError("beta_min, beta_max y niveles deben tener la misma longitud.")

    if np.any(beta_min < 0.0):
        raise ValueError("Todos los valores de beta_min deben ser no negativos.")

    if np.any(beta_max < beta_min):
        raise ValueError("Debe cumplirse beta_max[r] >= beta_min[r] para todo r.")

    if np.any(niveles < 2):
        raise ValueError("Cada entrada de niveles debe ser al menos 2.")

    grids_beta = [
        np.linspace(beta_min[r], beta_max[r], niveles[r])
        for r in range(len(beta_min))
    ]

    candidatos = []

    for beta in product(*grids_beta):
        beta = np.asarray(beta, dtype=float)

        # Evitamos el contrato idénticamente cero.
        if np.sum(beta) > 0.0:
            candidatos.append(beta)

    return candidatos


# ============================================================
# 3. Configuraciones listas para correr
# ============================================================

def crear_problema_rapido():
    """
    Problema pequeño para pruebas rápidas.

    parametros_agente contiene los parámetros del problema HJB del agente:
    T: horizonte temporal
    dtau: paso temporal usado en la ecuación HJB
    gamma: aversión al riesgo del agente
    eta: penalización por velocidad de ejecución
    sigma: volatilidad del precio
    rho, kappa, lambda: parámetros dependientes del régimen
    A: penalización terminal por inventario remanente
    q_max: velocidad máxima de ejecución permitida
    n_q_candidates: número de controles discretos usados en la búsqueda de q

    parametros_cliente contiene los parámetros usados en la simulación Monte Carlo:
    gamma_C: aversión al riesgo del cliente
    P0, I0, Q0: estado inicial
    regime0: régimen inicial
    n_paths: número de trayectorias simuladas
    seed0: semilla inicial
    silence_hjb: si True, oculta los mensajes del solver HJB
    """
    n_puntos = 5

    grids = (
        np.linspace(0.0, 5.0, n_puntos),
        np.linspace(75.0, 85.0, n_puntos),
        np.linspace(0.0, 1.0, n_puntos),
    )

    parametros_agente = {
        "T": 1.0,
        "dtau": 0.1,
        "gamma": 0.015,
        "eta": 1.5,
        "sigma": 1.0,
        "rho": [2.0, 0.8],
        "kappa": [0.8, 2.5],
        "lambda": [2.0, 3.0],
        "A": 115.0,
        "q_max": 10.0,
        "n_q_candidates": 100,
        "tol": 1e-6,
        "max_iter": 20,
        "print_m_matrix_warnings": False,
    }

    parametros_cliente = {
        "gamma_C": 0.015,
        "P0": 80.0,
        "I0": 0.0,
        "Q0": 1.0,
        "regime0": 0,
        "n_paths": 300,
        "seed0": 2026,
        "silence_hjb": True,
    }

    return grids, parametros_agente, parametros_cliente


def crear_problema_base():
    """
    Problema más fino. Es más lento que crear_problema_rapido().
    """
    n_puntos = 10

    grids = (
        np.linspace(0.0, 5.0, n_puntos),
        np.linspace(75.0, 85.0, n_puntos),
        np.linspace(0.0, 1.0, n_puntos),
    )

    parametros_agente = {
        "T": 1.0,
        "dtau": 0.05,
        "gamma": 0.015,
        "eta": 1.5,
        "sigma": 1.0,
        "rho": [2.0, 0.8],
        "kappa": [0.8, 2.5],
        "lambda": [2.0, 3.0],
        "A": 115.0,
        "q_max": 10.0,
        "n_q_candidates": 100,
        "tol": 1e-6,
        "max_iter": 20,
        "print_m_matrix_warnings": False,
    }

    parametros_cliente = {
        "gamma_C": 0.015,
        "P0": 80.0,
        "I0": 0.0,
        "Q0": 1.0,
        "regime0": 0,
        "n_paths": 300,
        "seed0": 2026,
        "silence_hjb": True,
    }

    return grids, parametros_agente, parametros_cliente


# ============================================================
# 4. Evaluar un contrato dado
# ============================================================

def simular_trayectorias(resultado_hjb, grids, parametros_agente, contrato, parametros_cliente):
    """
    Simula n_paths trayectorias bajo la política óptima inducida por el contrato.
    """
    n_paths = int(parametros_cliente["n_paths"]) 
    seed0 = int(parametros_cliente["seed0"])

    paths = []

    for s in range(n_paths):
        path = hjb.simulate_one_path(
            result=resultado_hjb,
            grids=grids,
            params=parametros_agente,
            contract_func=contrato,
            seed=seed0 + s,
            I0=parametros_cliente["I0"],
            P0=parametros_cliente["P0"],
            Q0=parametros_cliente["Q0"],
            regime0=parametros_cliente["regime0"],
        )
        paths.append(path)

    return paths


def calcular_metricas_cliente(paths, contrato, parametros_agente, parametros_cliente):
    """
    Calcula C0*, J y métricas auxiliares a partir de las trayectorias simuladas.
    """
    gamma_C = float(parametros_cliente["gamma_C"])
    gamma_D = float(parametros_agente["gamma"])
    P0 = float(parametros_cliente["P0"])
    A = float(parametros_agente["A"])
    dt = float(parametros_agente["dtau"])

    pagos_variables = []
    riqueza_agente_sin_C0 = []
    inventarios_terminales = []
    conteos_clip = []

    for path in paths:
        # Para cada trayectoria, tomamos los extremos izquierdos de las mallas en t y P, y calculamos c(t) en esos puntos.
        t_left = np.asarray(path["t"][:-1], dtype=float)
        P_left = np.asarray(path["P"][:-1], dtype=float)
        c_left = np.asarray(contrato(t_left), dtype=float)

        # Pago variable del contrato: integral P_t c_t dt. Se aproxima mediante suma de Riemann en los extremos izquierdos.
        pago_variable = float(np.sum(P_left * c_left) * dt)

        # Valores terminales de Q, P, X.
        Q_T = float(path["Q"][-1])
        P_T = float(path["P"][-1])
        X_T = float(path["X"][-1])

        # Riqueza terminal del agente antes del pago fijo C0.
        # Incluye riqueza acumulada X_T,
        # y penalización cuadrática por inventario terminal.
        Y_T = X_T - P_T * Q_T - A * Q_T**2

        pagos_variables.append(pago_variable)
        riqueza_agente_sin_C0.append(Y_T)
        inventarios_terminales.append(Q_T)
        conteos_clip.append(float(path.get("clip_count", 0)))

    pagos_variables = np.asarray(pagos_variables, dtype=float)
    riqueza_agente_sin_C0 = np.asarray(riqueza_agente_sin_C0, dtype=float)
    inventarios_terminales = np.asarray(inventarios_terminales, dtype=float)
    conteos_clip = np.asarray(conteos_clip, dtype=float)

    # Restricción de participación del agente:
    # C0 >= (1/gamma_D) log E[exp(-gamma_D Y_T)].
    C0_sin_restriccion = (1.0 / gamma_D) * logmeanexp(-gamma_D * riqueza_agente_sin_C0)
    C0_estrella = max(0.0, C0_sin_restriccion)

    # Objetivo del cliente:
    # J = C0* - P0 + (1/gamma_C) log E[exp(gamma_C int P_t c_t dt)].
    pago_ajustado_por_riesgo = (1.0 / gamma_C) * logmeanexp(gamma_C * pagos_variables)
    J = C0_estrella - P0 + pago_ajustado_por_riesgo

    # Costo de implementación del contrato para el cliente
    # pago fijo + pago variable - valor inicial de referencia
    IS = C0_estrella + pagos_variables - P0

    return {
        "objective_J": float(J),
        "C0_star": float(C0_estrella),
        "mean_IS": float(np.mean(IS)),
        "mean_variable_payment": float(np.mean(pagos_variables)),
        "mean_agent_YT": float(np.mean(riqueza_agente_sin_C0)),
        "mean_terminal_inventory": float(np.mean(inventarios_terminales)),
        "mean_clip_count": float(np.mean(conteos_clip)),
    }


def evaluar_contrato(contrato, grids, parametros_agente, parametros_cliente, beta=None):
    """
    Evalúa un contrato dado desde el punto de vista del cliente.

    Pasos:
        1. Resuelve la HJB del agente inducida por el contrato.
        2. Simula trayectorias bajo la política óptima.
        3. Calcula C0*, J y métricas auxiliares del cliente.

    Si el contrato proviene de una parametrización Bernstein, se puede pasar
    beta para que quede guardado en la salida.
    """
    inicio_hjb = time.perf_counter()
    with silenciar_si(parametros_cliente.get("silence_hjb", True)):
        resultado_hjb = hjb.solve_hjb(grids, parametros_agente, contrato)
    segundos_hjb = time.perf_counter() - inicio_hjb

    inicio_mc = time.perf_counter()
    paths = simular_trayectorias(resultado_hjb, grids, parametros_agente, contrato, parametros_cliente)
    segundos_mc = time.perf_counter() - inicio_mc

    metricas = calcular_metricas_cliente(paths, contrato, parametros_agente, parametros_cliente)
    metricas["hjb_seconds"] = float(segundos_hjb)
    metricas["mc_seconds"] = float(segundos_mc)

    salida = {
        "contrato": contrato,
        "resultado_hjb": resultado_hjb,
        "paths": paths,
        "metricas": metricas,
    }

    if beta is not None:
        salida["beta"] = np.asarray(beta, dtype=float)

    return salida


def evaluar_beta(beta, grids, parametros_agente, parametros_cliente):
    """
    Evalúa un candidato beta de la familia Bernstein.
    """
    beta = np.asarray(beta, dtype=float)
    contrato = contrato_bernstein(beta, T=parametros_agente["T"])
    return evaluar_contrato(
        contrato=contrato,
        grids=grids,
        parametros_agente=parametros_agente,
        parametros_cliente=parametros_cliente,
        beta=beta,
    )


# ============================================================
# 5. Búsqueda sobre una malla finita de contratos
# ============================================================

def buscar_mejor_contrato(beta_candidates, grids, parametros_agente, parametros_cliente, imprimir=True):
    """
    Evalúa una lista de candidatos beta y escoge el de menor J.
    Regresa:
    tabla: DataFrame con las métricas de todos los candidatos.
    mejor: diccionario completo del mejor candidato, incluyendo beta,
           contrato, solución HJB, trayectorias simuladas y métricas.
    """
    filas = []
    mejor = None

    for idx, beta in enumerate(beta_candidates, start=1):
        evaluacion = evaluar_beta(beta, grids, parametros_agente, parametros_cliente)
        metricas = evaluacion["metricas"]

        fila = {
            "candidate": idx,
            "objective_J": metricas["objective_J"],
            "C0_star": metricas["C0_star"],
            "mean_IS": metricas["mean_IS"],
            "mean_variable_payment": metricas["mean_variable_payment"],
            "mean_agent_YT": metricas["mean_agent_YT"],
            "mean_terminal_inventory": metricas["mean_terminal_inventory"],
            "mean_clip_count": metricas["mean_clip_count"],
            "hjb_seconds": metricas["hjb_seconds"],
            "mc_seconds": metricas["mc_seconds"],
        }

        for r, beta_r in enumerate(evaluacion["beta"]):
            fila[f"beta_{r}"] = float(beta_r)

        filas.append(fila)

        if mejor is None or metricas["objective_J"] < mejor["metricas"]["objective_J"]:
            mejor = evaluacion

        if imprimir:
            print(
                f"[{idx:03d}] beta={np.round(evaluacion['beta'], 4)} | "
                f"J={metricas['objective_J']:.6f} | "
                f"C0*={metricas['C0_star']:.6f} | "
                f"Q_T medio={metricas['mean_terminal_inventory']:.4f}"
            )

    tabla = pd.DataFrame(filas).sort_values("objective_J", ascending=True).reset_index(drop=True)
    return tabla, mejor


# ============================================================
# 6. Demo ejecutable desde terminal
# ============================================================

def correr_demo_rapido():
    """
    Corre una prueba pequeña de grado 2 con integral normalizada a 1.
    """
    grids, parametros_agente, parametros_cliente = crear_problema_rapido()

    # Candidatos normalizados
    # beta_candidates = malla_beta_simplex(
    #     grado=2,
    #     niveles=7,
    #     T=parametros_agente["T"],
    #     integral_objetivo=1.0,
    # )

    # Candidatos sin normalizar.
    beta_candidates = malla_beta_caja(
        beta_min=[0.0, 0.0, 0.0],
        beta_max=[2.0, 2.0, 2.0],
        niveles=[7, 7, 7],
    )

    print(f"Evaluando {len(beta_candidates)} contratos Bernstein grado 2...")
    tabla, mejor = buscar_mejor_contrato(
        beta_candidates=beta_candidates,
        grids=grids,
        parametros_agente=parametros_agente,
        parametros_cliente=parametros_cliente,
    )

    print("\nMejor contrato encontrado")
    print("beta* =", np.round(mejor["beta"], 6))
    print("J*     =", mejor["metricas"]["objective_J"])
    print("C0*    =", mejor["metricas"]["C0_star"])

    return tabla, mejor


if __name__ == "__main__":
    tabla, mejor = correr_demo_rapido()
    tabla.to_csv("resultados_grid_cliente_sencillo.csv", index=False)
    print("\nTabla guardada en resultados_grid_cliente_sencillo.csv")
