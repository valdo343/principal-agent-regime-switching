"""
Runner simple con cache para una sola familia de contratos.

Objetivo:
- Se define una familia de contratos con un nombre.
- Para cada contrato se resuelve el problema del agente con hjb_policy_iteration.solve_hjb.
- Se simulan trayectorias usando la estrategia óptima del agente.
- Se guardan los resultados en un archivo .pkl para no recalcular las HJB correspondientes a los contratos.
- Si el .pkl ya existe, se carga y se pueden cambiar las gráficas sin volver a resolver la HJB.

Este archivo debe estar en la misma carpeta que hjb_policy_iteration.py.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import hjb_policy_iteration as hjb


Contract = Callable[[np.ndarray], np.ndarray]
Record = Dict[str, Any]


# ============================================================
# 1. Familia de contratos
# ============================================================
# Para usar otra familia, cambiar FAMILY_NAME, las funciones de contrato
# y el diccionario que regresa get_contracts().

FAMILY_NAME = "Nombre de la familia"


def contract_1(t):
    t = np.asarray(t, dtype=float)
    return (1 - t) ** 2 + 2 * 1.5 * t * (1 - t) + 0.5 * t**2


def contract_2(t):
    t = np.asarray(t, dtype=float)
    return 0.923 * (1 - t) ** 2 + 2 * 1.615 * t * (1 - t) + 0.462 * t**2


def contract_3(t):
    t = np.asarray(t, dtype=float)
    return 1.071 * (1 - t) ** 2 + 2 * 1.286 * t * (1 - t) + 0.643 * t**2


def contract_4(t):
    t = np.asarray(t, dtype=float)
    return 1.154 * (1 - t) ** 2 + 2 * 1.154 * t * (1 - t) + 0.692 * t**2


def get_contracts() -> Dict[str, Contract]:
    """
    Regresa los contratos de la familia correspondiente.

    Las llaves son los nombres que aparecerán en consola, gráficas y cache.
    """
    return {
        "Contrato 1": contract_1,
        "Contrato 2": contract_2,
        "Contrato 3": contract_3,
        "Contrato 4": contract_4,
    }


# Textos para títulos de las gráficas
CONTRACT_TITLES = {
    "Contrato 1": "Contrato de escala fija 1",
    "Contrato 2": "Contrato de escala fija 2",
    "Contrato 3": "Contrato de escala fija 3",
    "Contrato 4": "Contrato de escala fija 4",
}


# ============================================================
# 2. Configuración base
# ============================================================

def make_grids(n_points: int = 10):
    I_grid = np.linspace(0.0, 5.0, n_points)
    P_grid = np.linspace(75.0, 85.0, n_points)
    Q_grid = np.linspace(0.0, 1.0, n_points)
    return I_grid, P_grid, Q_grid


def make_params() -> Dict[str, object]:
    return {
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


# ============================================================
# 3. Cache pkl
# ============================================================

def _slug(text: str) -> str:
    """Convierte un nombre en una etiqueta simple para nombres de archivo."""
    return (
        text.lower()
        .replace(" ", "_")
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
    )


def _cacheable_records(records: List[Record]) -> List[Record]:
    """
    Hace una copia de los registros sin guardar las funciones de contrato.

    Evitamos guardar objetos función porque pueden dar problemas al cargar el .pkl
    en otra sesión. Al cargar, se vuelven a asociar usando attach_contract_functions().
    """
    cache: List[Record] = []

    for record in records:
        clean_record = dict(record)
        clean_record.pop("contract_func", None)
        cache.append(clean_record)

    return cache


def attach_contract_functions(records: List[Record], contracts: Dict[str, Contract]) -> List[Record]:
    """Reasigna contract_func a cada registro usando el nombre del contrato."""
    for record in records:
        name = str(record["name"])

        if name not in contracts:
            raise KeyError(
                f"El contrato {name!r} está en el pkl, pero no está definido "
                "en get_contracts()."
            )

        record["contract_func"] = contracts[name]

    return records


def save_records_to_pkl(records: List[Record], family_name: str, pkl_path: Path) -> None:
    """Guarda la familia y sus resultados en un archivo .pkl."""
    pkl_path.parent.mkdir(parents=True, exist_ok=True)

    cache_data = {
        "family_name": family_name,
        "records": _cacheable_records(records),
    }

    with open(pkl_path, "wb") as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Resultados guardados en: {pkl_path}")


def load_records_from_pkl(
    pkl_path: Path,
    contracts: Dict[str, Contract],
    expected_family_name: Optional[str] = None,
) -> List[Record]:
    """Carga resultados desde pkl y vuelve a asociar las funciones de contrato."""
    with open(pkl_path, "rb") as f:
        cache_data = pickle.load(f)

    family_name = cache_data["family_name"]
    records = cache_data["records"]

    if expected_family_name is not None and family_name != expected_family_name:
        raise ValueError(
            f"El pkl corresponde a la familia {family_name!r}, "
            f"pero se esperaba {expected_family_name!r}."
        )

    records = attach_contract_functions(records, contracts)

    print(f"Resultados cargados desde: {pkl_path}")
    print("No se recalculó solve_hjb.")

    return records


# ============================================================
# 4. Resolver HJB y simular contratos
# ============================================================

def run_contracts(
    family_name: str,
    contracts: Dict[str, Contract],
    grids,
    params: Dict[str, object],
    n_sims: int = 4,
    base_seed: int = 100,
    I0: float = 0.0,
    P0: float = 80.0,
    Q0: float = 1.0,
    regime0: int = 0,
) -> List[Record]:
    """
    Resuelve HJB + simulaciones para todos los contratos de la familia.

    Para comparar contratos con el mismo ruido, la simulación s usa la semilla base_seed + s en todos los contratos.
    """
    records: List[Record] = []
    n_contracts = len(contracts)

    for contract_idx, (name, contract_func) in enumerate(contracts.items(), start=1):
        print("\n" + "=" * 80)
        print(f"Familia: {family_name} | Contrato {contract_idx}/{n_contracts}: {name}")
        print("=" * 80)

        # Resolver HJB para el contracto actual y guardar en result
        result = hjb.solve_hjb(
            grids=grids,
            params=params,
            contract_func=contract_func,
        )

        # Simular n_sims trayectorias usando la estrategia óptima del agente, y guardarlas en paths
        paths = [
            hjb.simulate_one_path(
                result=result,
                grids=grids,
                params=params,
                contract_func=contract_func,
                seed=base_seed + s,
                I0=I0,
                P0=P0,
                Q0=Q0,
                regime0=regime0,
            )
            for s in range(n_sims)
        ]

        # Guardamos resultados en un registro para esta familia y contrato.
        record: Record = {
            "family": family_name,
            "name": name,
            "contract_func": contract_func,
            "params": params.copy(),
            "grids": grids,
            "result": result,
            "paths": paths,
            "seeds": [base_seed + s for s in range(n_sims)],
            "clip_counts": [path["clip_count"] for path in paths],
        }

        # Guardamos el registro "record" en la lista de registros "records".
        records.append(record)

        print("Semillas usadas:", record["seeds"])
        print("Clippings fuera de malla:", record["clip_counts"])

    return records


# ============================================================
# 5. Re-simulación sin recalcular HJB
# ============================================================

def resimulate_records_with_common_seeds(
    records: List[Record],
    n_sims: int = 4,
    base_seed: int = 100,
    I0: float = 0.0,
    P0: float = 80.0,
    Q0: float = 1.0,
    regime0: int = 0,
) -> List[Record]:
    """
    Rehace SOLO las simulaciones usando las estrategias óptimas ya guardadas.

    Esta función no llama solve_hjb.
    """
    for record in records:
        result = record["result"]
        grids = record["grids"]
        params = record["params"]
        contract_func = record["contract_func"]

        paths = [
            hjb.simulate_one_path(
                result=result,
                grids=grids,
                params=params,
                contract_func=contract_func,
                seed=base_seed + s,
                I0=I0,
                P0=P0,
                Q0=Q0,
                regime0=regime0,
            )
            for s in range(n_sims)
        ]

        record["paths"] = paths
        record["seeds"] = [base_seed + s for s in range(n_sims)]
        record["clip_counts"] = [path["clip_count"] for path in paths]

    return records


# ============================================================
# 6. Gráficas
# ============================================================

def _display_contract_name(contract_name: str) -> str:
    return CONTRACT_TITLES.get(contract_name, contract_name)

def _maybe_save(fig, save_path: Optional[Path]) -> None:
    """Función auxiliar para guardar figuras localmente."""
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figura guardada en: {save_path}")

def plot_resumen_contrato_2x2(
    records: List[Record],
    family_name: str,
    contract_index: int = 0,
    sim_indices: Optional[List[int]] = None,
    show_legend: bool = True,
    regime_offset: float = 0.0,
    save_path: Optional[Path] = None,
    show: bool = True,
):
    """
    Grafica un solo contrato en panel 2x2, superponiendo varias simulaciones.

    Paneles:
        (1,1) Función de ponderación c(t)
        (1,2) Estrategia de ejecución del agente q*(t)
        (2,1) Impacto en el precio
        (2,2) Cambios de régimen

    Parámetros:
        records: lista de registros de una familia.
        family_name: nombre de la familia de contratos.
        contract_index: índice del contrato dentro de records.
        sim_indices: simulaciones que se quieren graficar.
                     Si None, grafica todas las simulaciones disponibles.
        regime_offset: desplazamiento vertical pequeño para separar visualmente las curvas de régimen. Para no desplazar, usar 0.0.
    """
    if contract_index < 0 or contract_index >= len(records):
        raise IndexError(
            f"contract_index={contract_index} fuera de rango. "
            f"Hay {len(records)} contratos en esta familia."
        )

    record = records[contract_index]

    name = str(record["name"])
    contract_func = record["contract_func"]
    params = record["params"]
    paths = record["paths"]

    T = float(params["T"])
    n_paths = len(paths)

    if n_paths == 0:
        raise ValueError(f"El contrato {name} no tiene trayectorias simuladas.")

    if sim_indices is None:
        sim_indices = list(range(n_paths))

    for s in sim_indices:
        if s < 0 or s >= n_paths:
            raise IndexError(
                f"sim_index={s} fuera de rango para {name}. "
                f"Este contrato tiene {n_paths} simulaciones."
            )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))


    # ============================================================
    # (1,1) Función de ponderación
    # ============================================================
    ax = axes[0, 0]

    t_contract = np.linspace(0.0, T, 300)
    c_contract = contract_func(t_contract)

    ax.plot(t_contract, c_contract, linewidth=2.4)
    ax.set_title("Contrato")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$c(t)$")
    ax.grid(True, alpha=0.3)


    # ============================================================
    # (1,2) Estrategia de ejecución del agente
    # ============================================================
    ax = axes[0, 1]

    for s in sim_indices:
        path = paths[s]
        t = np.asarray(path["t"], dtype=float)
        q = np.asarray(path["q"], dtype=float)

        if len(q) == len(t) - 1:
            t_q = t[:-1]
        elif len(q) == len(t):
            t_q = t
        else:
            t_q = t[:len(q)]

        ax.plot(t_q, q, linewidth=2.0, alpha=0.85, label=f"Sim {s}")

    ax.set_title("Estrategia de ejecución del agente")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$q_t^*$")
    ax.grid(True, alpha=0.3)

    if show_legend:
        ax.legend()

    # ============================================================
    # (2,1) Impacto en el precio
    # ============================================================
    ax = axes[1, 0]

    ylabel = r"$I_t$"

    for s in sim_indices:
        path = paths[s]
        t = np.asarray(path["t"], dtype=float)
        I = np.asarray(path["I"], dtype=float)

        t_I = t[:len(I)]

        ax.plot(t_I, I, linewidth=2.0, alpha=0.85, label=f"Sim {s}")

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.35)
    ax.set_title("Impacto en el precio")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    if show_legend:
        ax.legend()

    # ============================================================
    # (2,2) Cambios de régimen
    # ============================================================
    ax = axes[1, 1]

    for k, s in enumerate(sim_indices):
        path = paths[s]
        t = np.asarray(path["t"], dtype=float)
        regime = np.asarray(path["regime"], dtype=float)

        if len(regime) == len(t) - 1:
            t_reg = t[:-1]
        elif len(regime) == len(t):
            t_reg = t
        else:
            t_reg = t[:len(regime)]

        regime_plot = regime + k * regime_offset

        ax.step(
            t_reg,
            regime_plot,
            where="post",
            linewidth=2.0,
            alpha=0.85,
            label=f"Sim {s}",
        )

    ax.set_title("Cambios de régimen")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("Régimen")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["0", "1"])
    ax.grid(True, alpha=0.3)

    if regime_offset > 0.0:
        ax.set_ylim(-0.05, 1.0 + regime_offset * len(sim_indices) + 0.05)

    if show_legend:
        ax.legend()

    plt.tight_layout()
    _maybe_save(fig, save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_resumenes_2x2_por_contrato(
    records: List[Record],
    family_name: str,
    output_dir: Path,
    sim_indices: Optional[List[int]] = None,
    save_figures: bool = True,
    show_figures: bool = True,
):
    """
    Genera una figura 2x2 para cada contrato de la familia.

    Si sim_indices=None, grafica todas las simulaciones disponibles
    para cada contrato.
    """
    family_slug = _slug(family_name)

    for contract_index, record in enumerate(records):
        contract_name = str(record["name"])
        contract_slug = _slug(contract_name)

        save_path = None
        if save_figures:
            save_path = output_dir / f"resumen_2x2_{family_slug}_{contract_slug}.png"

        plot_resumen_contrato_2x2(
            records=records,
            family_name=family_name,
            contract_index=contract_index,
            sim_indices=sim_indices,
            regime_offset=0.0,
            save_path=save_path,
            show=show_figures,
        )

def _make_axes_grid(n_items: int, title: str):
    """Crea una malla de subgráficas simple para cualquier número de contratos."""
    if n_items <= 0:
        raise ValueError("Se necesita al menos un contrato para graficar.")

    n_cols = 2 if n_items > 1 else 1
    n_rows = int(np.ceil(n_items / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(8 * n_cols, 5.5 * n_rows),
        sharex=True,
        squeeze=False,
    )
    fig.suptitle(title, fontsize=20)

    return fig, axes.ravel()


def plot_contract_shapes(
    records: List[Record],
    family_name: str,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """Grafica la forma c(t) de todos los contratos de la familia."""
    params = records[0]["params"]
    T = float(params["T"])
    t_plot = np.linspace(0.0, T, 300)

    fig, axes = _make_axes_grid(
        n_items=len(records),
        title=f"Formas de contrato - {family_name}",
    )

    for ax, record in zip(axes, records):
        name = str(record["name"])
        contract_func = record["contract_func"]
        c_plot = contract_func(t_plot)

        ax.plot(t_plot, c_plot, linewidth=2.6)
        ax.set_title(_display_contract_name(name))
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$c(t)$")
        ax.grid(True, alpha=0.3)

    for ax in axes[len(records):]:
        ax.axis("off")

    plt.tight_layout()
    _maybe_save(fig, save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_q_strategies(
    records: List[Record],
    family_name: str,
    sim_index: Optional[int] = None,
    shade_regime: bool = True,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Grafica estrategias óptimas simuladas q*(t).

    sim_index=None dibuja todas las simulaciones disponibles por contrato.
    sim_index=0 dibuja una sola simulación por contrato.
    """
    if sim_index is None:
        title = f"Estrategias óptimas simuladas $q^*(t)$ - {family_name}"
    else:
        title = f"Estrategia óptima simulada $q^*(t)$ - {family_name} - simulación {sim_index}"

    fig, axes = _make_axes_grid(n_items=len(records), title=title)
    regime_bg_colors = {0: "tab:blue", 1: "tab:orange"}

    for ax, record in zip(axes, records):
        name = str(record["name"])
        paths = record["paths"]

        if sim_index is None:
            paths_to_plot = paths
        else:
            if sim_index < 0 or sim_index >= len(paths):
                raise IndexError(f"sim_index={sim_index} fuera de rango para {name}.")
            paths_to_plot = [paths[sim_index]]

        for path in paths_to_plot:
            t = np.asarray(path["t"])
            q = np.asarray(path["q"])
            regime = np.asarray(path["regime"])

            if sim_index is not None and shade_regime:
                regime_interval = regime[:-1] if len(regime) == len(t) else regime

                for n in range(len(q)):
                    reg = int(regime_interval[n])
                    ax.axvspan(
                        t[n],
                        t[n + 1],
                        color=regime_bg_colors[reg],
                        alpha=0.10,
                        linewidth=0,
                    )

                ax.plot(t[:-1], q, color="black", linewidth=2.6)
            else:
                ax.plot(t[:-1], q, linewidth=2.2, alpha=0.80)

        ax.set_title(_display_contract_name(name))
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$q_t^*$")
        ax.grid(True, alpha=0.3)

    for ax in axes[len(records):]:
        ax.axis("off")

    if sim_index is not None and shade_regime:
        legend_handles = [
            Patch(facecolor="tab:blue", alpha=0.10, edgecolor="none", label="Régimen A"),
            Patch(facecolor="tab:orange", alpha=0.10, edgecolor="none", label="Régimen B"),
        ]
        fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=True)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
    else:
        plt.tight_layout()

    _maybe_save(fig, save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_price_impact(
    records: List[Record],
    family_name: str,
    sim_index: Optional[int] = None,
    plot_mean: bool = True,
    shade_regime: bool = False,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Grafica el impacto de precio generado por el agente.
    """
    if sim_index is None:
        title = f"Impacto de precio generado por el agente - {family_name}"
    else:
        title = f"Impacto de precio generado por el agente - {family_name} - simulación {sim_index}"

    fig, axes = _make_axes_grid(n_items=len(records), title=title)
    regime_bg_colors = {0: "tab:blue", 1: "tab:orange"}

    for ax, record in zip(axes, records):
        name = str(record["name"])
        paths = record["paths"]

        if sim_index is None:
            paths_to_plot = paths
        else:
            if sim_index < 0 or sim_index >= len(paths):
                raise IndexError(f"sim_index={sim_index} fuera de rango para {name}.")
            paths_to_plot = [paths[sim_index]]

        impact_curves = []
        t_last = None

        for path in paths_to_plot:
            t = np.asarray(path["t"])
            I = np.asarray(path["I"])
            t_last = t

            impact = I
            impact_curves.append(impact)

            if sim_index is not None and shade_regime:
                regime = np.asarray(path["regime"])
                regime_interval = regime[:-1] if len(regime) == len(t) else regime

                for n in range(len(t) - 1):
                    reg = int(regime_interval[n])
                    ax.axvspan(
                        t[n],
                        t[n + 1],
                        color=regime_bg_colors[reg],
                        alpha=0.10,
                        linewidth=0,
                    )

                ax.plot(t, impact, color="black", linewidth=2.6)
            else:
                ax.plot(t, impact, linewidth=2.0, alpha=0.75)

        if sim_index is None and plot_mean and len(impact_curves) > 1 and t_last is not None:
            impact_mean = np.mean(np.vstack(impact_curves), axis=0)
            ax.plot(t_last, impact_mean, color="black", linewidth=3.0, linestyle="--", label="Promedio")
            ax.legend()

        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.35)
        ax.set_title(_display_contract_name(name))
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$I_t$")
        ax.grid(True, alpha=0.3)

    for ax in axes[len(records):]:
        ax.axis("off")

    if sim_index is not None and shade_regime:
        legend_handles = [
            Patch(facecolor="tab:blue", alpha=0.10, edgecolor="none", label="Régimen A"),
            Patch(facecolor="tab:orange", alpha=0.10, edgecolor="none", label="Régimen B"),
        ]
        fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=True)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
    else:
        plt.tight_layout()

    _maybe_save(fig, save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_full_simulations_per_contract(
    records: List[Record],
    family_name: str,
    show: bool = True,
) -> None:
    """Usa la gráfica 2x3 del módulo HJB para cada contrato."""
    for record in records:
        title = f"Simulaciones completas - {family_name} - {_display_contract_name(str(record['name']))}"
        hjb.plot_simulated_paths(record["paths"], title=title)

        if not show:
            plt.close("all")


def plot_all_results(
    records: List[Record],
    family_name: str,
    output_dir: Path,
    save_figures: bool = True,
    show_figures: bool = True,
    plot_full_simulations: bool = True,
    sim_index_for_single_path: Optional[int] = 0,
) -> None:
    """Grafica los resultados de una sola familia, sin recalcular HJB."""
    family_slug = _slug(family_name)

    save_contracts = output_dir / f"contratos_{family_slug}.png" if save_figures else None
    save_q_all = output_dir / f"estrategias_q_todas_{family_slug}.png" if save_figures else None
    save_impact_all = output_dir / f"impacto_precio_todas_{family_slug}.png" if save_figures else None

    save_q_one = None
    save_impact_one = None

    if save_figures and sim_index_for_single_path is not None:
        save_q_one = output_dir / f"estrategias_q_sim_{sim_index_for_single_path}_{family_slug}.png"
        save_impact_one = output_dir / f"impacto_precio_sim_{sim_index_for_single_path}_{family_slug}.png"

    plot_contract_shapes(
        records=records,
        family_name=family_name,
        save_path=save_contracts,
        show=show_figures,
    )

    plot_q_strategies(
        records=records,
        family_name=family_name,
        sim_index=None,
        save_path=save_q_all,
        show=show_figures,
    )

    plot_price_impact(
        records=records,
        family_name=family_name,
        sim_index=None,
        plot_mean=True,
        shade_regime=False,
        save_path=save_impact_all,
        show=show_figures,
    )

    if sim_index_for_single_path is not None:
        plot_q_strategies(
            records=records,
            family_name=family_name,
            sim_index=sim_index_for_single_path,
            shade_regime=True,
            save_path=save_q_one,
            show=show_figures,
        )

        plot_price_impact(
            records=records,
            family_name=family_name,
            sim_index=sim_index_for_single_path,
            plot_mean=False,
            shade_regime=True,
            save_path=save_impact_one,
            show=show_figures,
        )

        plot_resumenes_2x2_por_contrato(
            records=records,
            family_name=family_name,
            output_dir=output_dir,
            sim_indices=None,
            save_figures=save_figures,
            show_figures=show_figures,
        )

    if plot_full_simulations:
        plot_full_simulations_per_contract(
            records=records,
            family_name=family_name,
            show=show_figures,
        )


# ============================================================
# 7. Main con cache
# ============================================================

def main(
    family_name: str = FAMILY_NAME,
    contracts: Optional[Dict[str, Contract]] = None,
    n_points: int = 10,
    n_sims: int = 4,
    base_seed: int = 100,
    I0: float = 0.0,
    P0: float = 80.0,
    Q0: float = 1.0,
    regime0: int = 0,
    output_dir: Path = Path("figuras_contratos"),
    force_recompute: bool = False,
    resimulate_paths_when_loading: bool = False,
    save_after_resimulating: bool = True,
    make_plots: bool = True,
    save_figures: bool = True,
    show_figures: bool = True,
    plot_full_simulations: bool = True,
    sim_index_for_single_path: Optional[int] = 0,
) -> List[Record]:
    """
    Ejecuta la familia de contratos.

    Si existe el archivo .pkl y force_recompute=False, carga la estrategia guardada y no recalcula la HJB.
    
    Para recalcular desde cero, usar force_recompute=True.
    """
    if contracts is None:
        contracts = get_contracts()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    family_slug = _slug(family_name)
    pkl_path = output_dir / f"resultados_hjb_{family_slug}.pkl"

    if pkl_path.exists() and not force_recompute:
        records = load_records_from_pkl(
            pkl_path=pkl_path,
            contracts=contracts,
            expected_family_name=family_name,
        )

        if resimulate_paths_when_loading:
            print("Re-simulando trayectorias con semillas comunes, sin recalcular HJB...")
            records = resimulate_records_with_common_seeds(
                records=records,
                n_sims=n_sims,
                base_seed=base_seed,
                I0=I0,
                P0=P0,
                Q0=Q0,
                regime0=regime0,
            )

            if save_after_resimulating:
                save_records_to_pkl(records, family_name, pkl_path)
    else:
        print("No existe pkl o force_recompute=True. Calculando HJB desde cero...")
        grids = make_grids(n_points=n_points)
        params = make_params()

        records = run_contracts(
            family_name=family_name,
            contracts=contracts,
            grids=grids,
            params=params,
            n_sims=n_sims,
            base_seed=base_seed,
            I0=I0,
            P0=P0,
            Q0=Q0,
            regime0=regime0,
        )

        save_records_to_pkl(records, family_name, pkl_path)

    if make_plots:
        plot_all_results(
            records=records,
            family_name=family_name,
            output_dir=output_dir,
            save_figures=save_figures,
            show_figures=show_figures,
            plot_full_simulations=plot_full_simulations,
            sim_index_for_single_path=sim_index_for_single_path,
        )

    return records


if __name__ == "__main__":
    records = main()
