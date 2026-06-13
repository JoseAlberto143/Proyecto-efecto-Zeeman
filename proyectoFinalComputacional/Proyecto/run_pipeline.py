"""
=============================================================================
PIPELINE COMPLETO — Átomos tipo alcalino
=============================================================================

  1) Para cada átomo en la lista de trabajo:
        a) modo 'tabulated' (Li, Na): usa los parámetros Marinescu del paper
           original PRA 49, 982 (1994). Esos valores se mantienen porque sobre
           nuestra malla lineal el error sigue siendo < 500 cm⁻¹.
        b) modo 'refit' (K, Rb, Cs, y cualquier especie del catálogo): re-ajusta
           los 5 parámetros del pseudopotencial Marinescu-Dalgarno **por canal
           ℓ** (igual que la Tabla I del paper) contra los niveles NIST,
           pero sobre NUESTRA malla 100% lineal. Los valores tabulados originales
           fallan aquí porque fueron optimizados para una malla log-lineal que
           concentra puntos cerca del origen; los nuestros tienen h ~ 0.02 Bohr.
        c) modo 'hydrogen' (H): caso especial, potencial coulombiano puro.
        d) modo 'auto': elige según símbolo. Para iones (Ca+, B2+, ...) intenta
           refit con NIST embebido o descarga del NIST ASD.

  2) Resolver la ecuación radial para todos los pares (n, l) del subespacio
     {n0, n0+extra_n} usando Numerov + shooting + Wronskiano.
     Potencial: Marinescu-Dalgarno generalizado (5 parámetros + α_c).

  3) Calcular la constante de acoplamiento espín-órbita ξ_nℓ.

  4) Comparar contra los niveles experimentales NIST.

  5) Volcar todos los resultados a un CSV reutilizable.

  6) Generar tres familias de gráficas de validación:
        - V_eff(r) para cada átomo (incluye el "core" y la cola Coulomb)
        - P_nl(r) para los primeros estados de cada átomo
        - Errores E_calc - E_NIST por átomo


Salida CSV con columnas:
    symbol, Z, Z_c, n, l, term,
    E_calc_H, E_calc_eV, E_NIST_H, diff_cm-1,
    xi_nl_H, n_nodes, converged, method, fit_loss
=============================================================================
"""
from __future__ import annotations

import os
import csv
import math
import time
from dataclasses import asdict
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")        
import matplotlib.pyplot as plt

from alkali_radial_solver import (
    AlkaliAtom,
    ALKALI_GROUND_STATES,
    MARINESCU_PARAMS,
    NIST_REFERENCE_LEVELS,
    HARTREE_TO_EV,
    HARTREE_TO_CM,
    BOHR_TO_ANGSTROM,
    validate_against_nist,
    build_radial_mesh,
    find_bound_state,
    find_bound_state_near,
    compute_xi_nl,
)
from numerov_fast import warmup_jit, NUMBA_AVAILABLE


# =============================================================================
# CONFIGURACIÓN DEL PIPELINE
# =============================================================================
#
# Los alcalinos pesados (K, Rb, Cs) no se pueden resolver con precisión sobre
# una malla 100% lineal usando los parámetros Marinescu tabulados originales:
# esos valores asumen una malla logarítmica concentrada cerca del origen.
# Sobre nuestra malla lineal, los autovalores fluctúan miles de cm⁻¹ con la
# resolución porque a3 ~ −10..−24 produce un pozo demasiado singular.
#
# La solución correcta —recomendada por Wagner y consistente con la receta
# del paper— es reajustar los 5 parámetros (a1,a2,a3,a4,rc) por canal ℓ
# contra los niveles NIST en nuestra malla. Los nuevos parámetros se acotan
# para evitar pozos patológicos y los errores caen a ~10²–10³ cm⁻¹.
#
# Por defecto:
#   - Li, Na : parámetros Marinescu tabulados .
#   - K, Rb, Cs : reajuste por canal ℓ contra NIST.
#   - H : caso especial

DEFAULT_ATOMS_TABULATED  = ["Li", "Na"]
DEFAULT_ATOMS_NIST_REFIT = ["K", "Rb", "Cs"]
DEFAULT_ATOMS_QDT_FIT    = ["H"]

DEFAULT_ATOMS_FULL_CATALOG = (
    DEFAULT_ATOMS_QDT_FIT
    + DEFAULT_ATOMS_TABULATED
    + DEFAULT_ATOMS_NIST_REFIT
    + ["Fr", "Be+", "Mg+", "Ca+", "Sr+", "Ba+", "Ra+",
       "B2+", "Al2+", "Ga2+", "In2+", "Tl2+",
       "Cu", "Ag", "Au"]
)

ATOMIC_TERM_LETTER = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g", 5: "h"}


def term_label(n: int, l: int) -> str:
    """Genera la etiqueta espectroscópica nL para usar en CSV y gráficas."""
    letter = ATOMIC_TERM_LETTER.get(l, f"l={l}")
    return f"{n}{letter}"


def solve_state_fast(atom: AlkaliAtom, n: int, l: int, n_points: int = 200000):
    """Variante optimizada de `atom.solve_state` que usa los datos NIST embebidos
    como semilla cuando están disponibles, o una estimación QDT del defecto cuántico
    del mismo canal l cuando hay otros niveles tabulados con ese l.

    La malla radial se dimensiona para EL estado (n, l) solicitado, no para el
    máximo del subespacio. Esto permite que estados poco excitados conserven
    la resolución cerca del origen sin estirar r_max innecesariamente.
    """
    from alkali_radial_solver import RadialSolution, quantum_defect_from_energy
    r = build_radial_mesh(n_max=n, Z_c=atom.Z_c, n_points=n_points)
    V_func = lambda rr: atom.V_eff(rr, l)

    using_nist_seed = (n, l) in atom.nist_levels
    using_qdt_seed = False

    if using_nist_seed:
        E_rel = atom.nist_levels[(n, l)]
        E_guess = E_rel - atom.E_ionization_H
    else:
        
        same_l_defects = []
        for (nn, ll), E_rel in atom.nist_levels.items():
            if ll != l:
                continue
            E_abs = E_rel - atom.E_ionization_H
            d = quantum_defect_from_energy(nn, E_abs, atom.Z_c)
            if math.isfinite(d):
                same_l_defects.append(d)
        if same_l_defects:
            delta_l = float(np.mean(same_l_defects))
            E_guess = -(atom.Z_c ** 2) / (2.0 * (n - delta_l) ** 2)
            using_qdt_seed = True
        else:
            E_guess = -(atom.Z_c ** 2) / (2.0 * n * n)

    sol = find_bound_state_near(n, l, r, V_func, E_guess=E_guess,
                                window_frac=0.4, n_scan=25)

    has_seed = using_nist_seed or using_qdt_seed
    if (not sol.converged) or (not has_seed and sol.n_nodes != (n - l - 1)):
        
        E_max_search = -1e-5
        E_min_search = max(-(atom.Z_c ** 2) / 2.0 * 1.05, -1.0)  
        sol = find_bound_state(n, l, r, V_func, Z_c=atom.Z_c,
                               E_min=E_min_search, E_max=E_max_search, n_scan=200)
    if sol.converged:
        sol.xi_nl = compute_xi_nl(sol, lambda rr: atom.dV_eff(rr, l))
    return sol


# =============================================================================
# CONSTRUCCIÓN DE ÁTOMOS
# =============================================================================

def build_atom(symbol: str, mode: str = "auto",
               nist_levels: dict | None = None,
               verbose: bool = True) -> tuple[AlkaliAtom, str]:
    """Construye un AlkaliAtom usando la estrategia indicada.

    Args:
        symbol: símbolo de la especie ('H', 'Li', ..., 'Cs', 'Fr', 'Ca+', ...).
        mode:
            'tabulated'  – usar parámetros Marinescu de la tabla
                           (sólo Li, Na, K, Rb, Cs neutros).
            'refit'      – reajuste por canal ℓ contra NIST.
            'hydrogen'   – caso especial Coulomb puro (sólo H).
            'auto'       – elige automáticamente según `symbol`:
                            * H → hydrogen
                            * Li, Na → tabulated
                            * K, Rb, Cs → refit (los originales no convergen)
                            * cualquier otra del catálogo → refit con datos
                              NIST embebidos o descargados; si no hay datos,
                              fallback suave.
        nist_levels: si se pasa, se usan en lugar de los NIST embebidos.
        verbose: imprime progreso.

    Returns:
        (atom, method) con method ∈ {"tabulated", "nist_refit", "hydrogen_coulomb",
        "fallback_smooth"}.
    """
    if mode == "auto":
        if symbol == "H":
            mode = "hydrogen"
        elif symbol in DEFAULT_ATOMS_TABULATED:
            mode = "tabulated"
        else:
            mode = "refit"

    if mode == "hydrogen":
        if verbose:
            print(f"  -> Caso especial: Hidrógeno (potencial coulombiano puro)")
        gs = ALKALI_GROUND_STATES["H"]
        def E_rel_hydrogen(n: int) -> float:
            return 0.5 * (1.0 - 1.0 / (n * n))
        atom = AlkaliAtom(
            symbol="H", Z=gs["Z"], Z_c=1, n0=gs["n0"], l0=gs["l0"],
            params={"a1": 5.0, "a2": 1.0, "a3": 0.0, "a4": 0.0, "r_c": 1.0},
            is_tabulated=False, tabulated_params={},
            nist_levels=nist_levels or {
                (1, 0): E_rel_hydrogen(1),
                (2, 0): E_rel_hydrogen(2), (2, 1): E_rel_hydrogen(2),
                (3, 0): E_rel_hydrogen(3), (3, 1): E_rel_hydrogen(3),
                (3, 2): E_rel_hydrogen(3),
            },
            E_ionization_H=gs["E_ion_H"], alpha_c=0.0,
        )
        return atom, "hydrogen_coulomb"

    if mode == "tabulated":
        if symbol not in MARINESCU_PARAMS:
            raise ValueError(
                f"No hay parámetros Marinescu tabulados para {symbol}. "
                f"Use mode='refit' en su lugar."
            )
        if verbose:
            print(f"  -> Usando parámetros tabulados (Marinescu PRA 49, 982)")
        atom = AlkaliAtom.from_tabulated(symbol)
        if nist_levels is not None:
            atom.nist_levels = nist_levels
        return atom, "tabulated"

    if mode == "refit":
        if symbol in MARINESCU_PARAMS and symbol in ALKALI_GROUND_STATES:
            levels = nist_levels or NIST_REFERENCE_LEVELS.get(symbol, {})
            if not levels:
                raise ValueError(f"No hay datos NIST para refit de {symbol}.")
            if verbose:
                print(f"  -> Refit Marinescu por canal ℓ contra NIST "
                      f"({len(levels)} niveles)")
            atom = AlkaliAtom.from_nist_fit(symbol, levels_relative=levels,
                                            verbose=verbose, per_l=True)
            return atom, "nist_refit"

        from periodic_catalog import get_species
        try:
            sp = get_species(symbol)
        except ValueError as e:
            raise ValueError(
                f"Símbolo '{symbol}' no está en el catálogo periódico. {e}"
            ) from None

        levels = nist_levels or NIST_REFERENCE_LEVELS.get(symbol, {})
        if not levels:
            try:
                from nist_extractor import get_levels_for_element
                elem = symbol.rstrip("+")
                if "+" in symbol:
                    nplus = symbol.count("+")
                    if symbol[-2:-1].isdigit():
                        nplus = int(symbol[-2])
                    ion_stage = 1 + nplus
                    elem = "".join(c for c in elem if c.isalpha())
                else:
                    ion_stage = 1
                levels = get_levels_for_element(
                    elem, ionization_stage=ion_stage,
                    cache_dir="nist_cache", verbose=verbose,
                ) or {}
            except Exception as e:
                if verbose:
                    print(f"  [!] descarga NIST falló para {symbol}: {e}")
                levels = {}

        if levels:
            if verbose:
                print(f"  -> Refit Marinescu por canal ℓ para {symbol} "
                      f"({len(levels)} niveles NIST)")
            atom = AlkaliAtom.from_species(symbol, nist_levels=levels,
                                           verbose=verbose,
                                           prefer_tabulated=False)
            return atom, "nist_refit"

        if verbose:
            print(f"  [!] Sin datos NIST para {symbol}: usando parámetros suaves.")
        atom = AlkaliAtom.from_species(symbol, verbose=verbose,
                                       prefer_tabulated=False)
        return atom, "fallback_smooth"

    raise ValueError(f"Modo de construcción '{mode}' no reconocido.")


# =============================================================================
# RESOLUCIÓN DE TODOS LOS ESTADOS Y CONSTRUCCIÓN DE FILAS CSV
# =============================================================================

def collect_atom_results(
    atom: AlkaliAtom,
    method: str,
    extra_n: int = 1,
) -> list[dict[str, Any]]:
    """Resuelve los estados del subespacio del proyecto y los presenta en
    una lista de diccionarios listos para volcar a CSV.

    Subespacio del proyecto:
        n ∈ {n0, n0+1, ..., n0+extra_n}
        l = 0, 1, ..., n-1
        (restricción del avance: si n == n0, l >= l0)
    """
    rows: list[dict[str, Any]] = []
    n0 = atom.n0
    nist = atom.nist_levels

    n_values = list(range(n0, n0 + 1 + extra_n))

    for n in n_values:
        for l in range(0, n):
            if n == n0 and l < atom.l0:
                continue
            try:
                sol = solve_state_fast(atom, n, l)
            except Exception as e:
                print(f"     [!] Error resolviendo {atom.symbol} (n={n},l={l}): {e}")
                continue

            if (n, l) in nist:
                E_nist_abs = nist[(n, l)] - atom.E_ionization_H
                diff_H = sol.E - E_nist_abs
                diff_cm = diff_H * HARTREE_TO_CM
            else:
                E_nist_abs = float("nan")
                diff_H = float("nan")
                diff_cm = float("nan")

            rows.append({
                "symbol":     atom.symbol,
                "Z":          atom.Z,
                "Z_c":        atom.Z_c,
                "n":          n,
                "l":          l,
                "term":       term_label(n, l),
                "E_calc_H":   sol.E,
                "E_calc_eV":  sol.E * HARTREE_TO_EV,
                "E_NIST_H":   E_nist_abs,
                "diff_cm-1":  diff_cm,
                "xi_nl_H":    sol.xi_nl,
                "n_nodes":    sol.n_nodes,
                "converged":  sol.converged,
                "method":     method,
                "fit_loss":   atom.fit_info.get("total_loss", "") if atom.fit_info else "",
            })

    return rows


# =============================================================================
# VOLCADO A CSV
# =============================================================================

def write_csv(rows: list[dict], path: str) -> None:
    """Escribe el listado de filas a CSV con cabecera consistente."""
    if not rows:
        print("[!] No hay filas para escribir.")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = {}
            for k, v in row.items():
                if isinstance(v, float):
                    if math.isnan(v):
                        formatted[k] = ""
                    elif abs(v) < 1e-3:
                        formatted[k] = f"{v:.6e}"
                    else:
                        formatted[k] = f"{v:.8f}"
                else:
                    formatted[k] = v
            writer.writerow(formatted)
    print(f"  CSV escrito: {path} ({len(rows)} filas)")


# =============================================================================
# GRÁFICAS DE VALIDACIÓN
# =============================================================================

def plot_potentials(atoms: dict[str, AlkaliAtom], out_path: str) -> None:
    """Grafica V_eff(r) para cada átomo, comparándolo con la cola Coulomb -Z_c/r."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=False)
    axes = axes.flatten()

    r_plot = np.linspace(0.05, 30.0, 500)

    for idx, (sym, atom) in enumerate(atoms.items()):
        if idx >= len(axes):
            break
        ax = axes[idx]
        V_l0 = atom.V_eff(r_plot, l=0)
        V_l1 = atom.V_eff(r_plot, l=1)
        V_coulomb = -atom.Z_c / r_plot
        V_full_coul = -atom.Z / r_plot

        ax.plot(r_plot, V_l0, "b-", lw=1.6, label=r"$V_{eff}(r)$, $\ell=0$")
        if not np.allclose(V_l0, V_l1):
            ax.plot(r_plot, V_l1, "g--", lw=1.2, label=r"$V_{eff}(r)$, $\ell=1$")
        ax.plot(r_plot, V_coulomb, "k:", lw=1.0,
                label=rf"$-Z_c/r$ (asíntota)")
        if atom.Z != atom.Z_c:
            ax.plot(r_plot, V_full_coul, "r:", lw=0.8, alpha=0.6,
                    label=rf"$-Z/r$ (núcleo)")
        ax.set_xlabel(r"$r$ (Bohr)")
        ax.set_ylabel(r"$V_{eff}(r)$ (Hartree)")
        ax.set_title(f"{sym} (Z={atom.Z}, $Z_c$={atom.Z_c})")
        ax.set_xlim(0, 25)
        V_at_1 = atom.V_eff(np.array([1.0]), l=0)[0]
        ax.set_ylim(min(-3.0, 1.2 * V_at_1), 0.05)
        ax.axhline(0, color="gray", lw=0.4)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)

    for j in range(len(atoms), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Pseudopotenciales efectivos (Marinescu-Dalgarno unificado)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Gráfica: {out_path}")


def plot_radial_functions(atoms: dict[str, AlkaliAtom], out_path: str) -> None:
    """Grafica P_nl(r) para los primeros estados del subespacio de cada átomo."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for idx, (sym, atom) in enumerate(atoms.items()):
        if idx >= len(axes):
            break
        ax = axes[idx]
        targets = []
        n0, l0 = atom.n0, atom.l0
        for (n, l) in [(n0, l0), (n0, l0+1), (n0+1, 0), (n0+1, 1)]:
            if l < n:
                targets.append((n, l))
        for (n, l) in targets:
            try:
                sol = solve_state_fast(atom, n, l)
                if not sol.converged:
                    continue
                mask = sol.r < min(60, 4*(n+2)**2)
                ax.plot(sol.r[mask], sol.P[mask], lw=1.2,
                        label=fr"$P_{{{term_label(n,l)}}}$  $E={sol.E:.4f}$ H")
            except Exception as e:
                continue
        ax.set_xlabel(r"$r$ (Bohr)")
        ax.set_ylabel(r"$P_{n\ell}(r) = r\cdot R_{n\ell}(r)$")
        ax.set_title(f"{sym}: funciones radiales reducidas")
        ax.axhline(0, color="gray", lw=0.4)
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)

    for j in range(len(atoms), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Funciones radiales reducidas $P_{n\\ell}(r)$ (Numerov + shooting)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Gráfica: {out_path}")


def plot_validation_errors(rows: list[dict], out_path: str) -> None:
    """Diagrama de barras: error relativo porcentual (E_calc − E_NIST)/E_NIST × 100
    por estado y átomo. La métrica relativa es preferible a cm⁻¹ porque no
    castiga visualmente a los niveles profundos (que tienen energías
    absolutas grandes) y refleja la desviación real del modelo empírico
    respecto al experimento.
    """
    by_symbol: dict[str, list[tuple[str, float]]] = {}
    for r in rows:
        if r["E_NIST_H"] == "" or r["E_calc_H"] == "":
            continue
        try:
            E_nist = float(r["E_NIST_H"])
            E_calc = float(r["E_calc_H"])
        except (ValueError, TypeError):
            continue
        if not math.isfinite(E_nist) or not math.isfinite(E_calc) or E_nist == 0.0:
            continue
        pct = (E_calc - E_nist) / abs(E_nist) * 100.0
        by_symbol.setdefault(r["symbol"], []).append((r["term"], pct))

    if not by_symbol:
        print("[!] Sin datos NIST suficientes para graficar errores.")
        return

    n_symbols = len(by_symbol)
    fig, axes = plt.subplots(
        nrows=(n_symbols + 1) // 2, ncols=2,
        figsize=(13, 3.0 * ((n_symbols + 1) // 2)),
        squeeze=False,
    )
    axes = axes.flatten()

    for idx, (sym, data) in enumerate(by_symbol.items()):
        ax = axes[idx]
        terms = [t for t, _ in data]
        pcts = [p for _, p in data]
        colors = ["tab:red" if abs(p) > 1.0 else "tab:blue" for p in pcts]
        bars = ax.bar(terms, pcts, color=colors, edgecolor="black", lw=0.5)
        ax.axhline(0, color="black", lw=0.7)
        ax.set_title(f"{sym}: error relativo (E_calc − E_NIST) / |E_NIST|")
        ax.set_ylabel(r"$\Delta E / |E_\mathrm{NIST}|$ (%)")
        ax.set_xlabel("Estado")
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=45)
        for bar, p in zip(bars, pcts):
            ax.annotate(f"{p:+.3f}%",
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3 if p >= 0 else -10),
                        textcoords="offset points",
                        ha="center", fontsize=7)

    for j in range(n_symbols, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Validación contra NIST: error relativo (%)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Gráfica: {out_path}")


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def run_pipeline(
    atoms_tabulated: list[str] | None = None,
    atoms_nist_refit: list[str] | None = None,
    atoms_qdt_fit:    list[str] | None = None,
    atoms_extra:      list[str] | None = None,
    out_dir: str = ".",
    extra_n: int = 1,
    verbose: bool = True,
) -> tuple[list[dict], dict[str, AlkaliAtom]]:
    """Ejecuta el pipeline completo y devuelve (rows, atoms_dict).

    Args:
        atoms_tabulated: especies a procesar con parámetros Marinescu de tabla.
            Defaults to Li, Na (errores ya pequeños en malla lineal).
        atoms_nist_refit: especies a procesar con re-ajuste por canal ℓ contra
            NIST sobre nuestra malla. Defaults to K, Rb, Cs.
        atoms_qdt_fit: especies a procesar con receta especial (sólo H por ahora).
        atoms_extra: especies adicionales del catálogo periódico (Fr, Ca+, ...).
            Se intenta refit con NIST embebido o descargado; si falla, fallback.
        out_dir: directorio de salida para CSV y PNGs.
        extra_n: cuántos n adicionales sobre n0 procesar (default 1 ⇒ n ∈ {n0, n0+1}).
        verbose: imprime progreso detallado.

    """
    if atoms_tabulated is None:
        atoms_tabulated = DEFAULT_ATOMS_TABULATED
    if atoms_nist_refit is None:
        atoms_nist_refit = DEFAULT_ATOMS_NIST_REFIT
    if atoms_qdt_fit is None:
        atoms_qdt_fit = DEFAULT_ATOMS_QDT_FIT
    if atoms_extra is None:
        atoms_extra = []

    os.makedirs(out_dir, exist_ok=True)
    all_rows: list[dict] = []
    atoms_dict: dict[str, AlkaliAtom] = {}

    if NUMBA_AVAILABLE:
        if verbose:
            print(f"[JIT] Calentando núcleo Numba...")
        t0 = time.time()
        warmup_jit()
        if verbose:
            print(f"[JIT] Listo en {time.time()-t0:.2f}s")

    plan: list[tuple[str, str]] = []
    seen: set[str] = set()
    for sym in atoms_qdt_fit:
        if sym not in seen:
            plan.append((sym, "hydrogen" if sym == "H" else "auto"))
            seen.add(sym)
    for sym in atoms_tabulated:
        if sym not in seen:
            plan.append((sym, "tabulated"))
            seen.add(sym)
    for sym in atoms_nist_refit:
        if sym not in seen:
            plan.append((sym, "refit"))
            seen.add(sym)
    for sym in atoms_extra:
        if sym not in seen:
            plan.append((sym, "auto"))
            seen.add(sym)

    for sym, mode in plan:
        print(f"\n[{sym}] " + "=" * 60)
        t0 = time.time()
        try:
            atom, method = build_atom(sym, mode=mode, verbose=verbose)
        except Exception as e:
            print(f"  [!] No se pudo construir {sym}: {e}")
            continue

        atoms_dict[sym] = atom
        rows = collect_atom_results(atom, method, extra_n=extra_n)
        all_rows.extend(rows)
        dt = time.time() - t0
        n_ok = sum(1 for r in rows if r["converged"])
        print(f"  -> {len(rows)} estados resueltos, {n_ok} convergidos, t={dt:.1f}s")

    # CSV
    print("\n" + "=" * 70)
    csv_path = os.path.join(out_dir, "alkali_levels.csv")
    write_csv(all_rows, csv_path)

    # Gráficas
    plot_potentials(atoms_dict,
                    os.path.join(out_dir, "plot_potentials.png"))
    plot_radial_functions(atoms_dict,
                          os.path.join(out_dir, "plot_radial_functions.png"))
    plot_validation_errors(all_rows,
                           os.path.join(out_dir, "plot_validation_errors.png"))

    # Resumen final
    print("\n" + "=" * 70)
    print("RESUMEN DE VALIDACIÓN")
    print("=" * 70)
    print(f"{'Atomo':<8} {'Método':<14} {'Estados':<8} {'Conv.':<6} "
          f"{'|Δ|_med (cm⁻¹)':<16} {'|Δ|_max (cm⁻¹)':<16}")
    by_sym: dict[str, list[float]] = {}
    by_sym_count: dict[str, list[int]] = {}
    by_sym_method: dict[str, str] = {}
    for r in all_rows:
        by_sym_count.setdefault(r["symbol"], [0, 0])
        by_sym_count[r["symbol"]][0] += 1
        if r["converged"]:
            by_sym_count[r["symbol"]][1] += 1
        by_sym_method[r["symbol"]] = r["method"]
        if isinstance(r["diff_cm-1"], float) and not math.isnan(r["diff_cm-1"]):
            by_sym.setdefault(r["symbol"], []).append(abs(r["diff_cm-1"]))
    for sym, (total, conv) in by_sym_count.items():
        diffs = by_sym.get(sym, [])
        meth = by_sym_method.get(sym, "?")
        if diffs:
            med = float(np.median(diffs))
            mx = float(np.max(diffs))
            print(f"{sym:<8} {meth:<14} {total:<8} {conv:<6} {med:<16.2f} {mx:<16.2f}")
        else:
            print(f"{sym:<8} {meth:<14} {total:<8} {conv:<6} {'n/a':<16} {'n/a':<16}")

    return all_rows, atoms_dict


if __name__ == "__main__":
    import sys
    if "--all" in sys.argv:
        rows, atoms = run_pipeline(out_dir="./out",
                                   atoms_extra=DEFAULT_ATOMS_FULL_CATALOG)
    else:
        rows, atoms = run_pipeline(out_dir="./out")
