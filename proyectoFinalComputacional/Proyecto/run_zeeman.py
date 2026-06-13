

"""
run_zeeman.py — Orquestador del efecto Zeeman sin diamagnetismo.

Versión Híbrida Exacta: 
1. Usa E y ξ estrictamente del alkali_levels.csv del usuario.
2. Reconstruye P(r) en un milisegundo usando Parametros.csv para calcular R real.
3. Permite todas las transiciones (sin asumir R=1).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import zeeman_matrix as zm
from alkali_radial_solver import AlkaliAtom, build_radial_mesh, find_bound_state_near
from periodic_catalog import get_species

HARTREE_TO_CM = zm.HARTREE_TO_CM
B_ATOMIC_TESLA = zm.B_ATOMIC_TESLA
BOHR_TO_NM = zm.BOHR_TO_NM
HC_ATOMIC = zm.HC_ATOMIC

# =============================================================================
# Carga de Datos
# =============================================================================

class CSVLevel:
    def __init__(self, E: float, xi_nl: float):
        self.E = E
        self.xi_nl = xi_nl

def load_levels_csv(path: str) -> dict[str, dict[tuple[int, int], CSVLevel]]:
    atoms: dict[str, dict[tuple[int, int], CSVLevel]] = {}
    invalid: dict[str, list[str]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            sym = row["symbol"]
            if row.get("converged", "True") != "True":
                continue
            E_str = row.get("E_calc_H", "").strip()
            if not E_str:
                continue
            E = float(E_str)
            if E >= 0.0:
                invalid.setdefault(sym, []).append(row.get("term", "?"))
                continue
            n, l = int(row["n"]), int(row["l"])
            xi = float(row["xi_nl_H"]) if row.get("xi_nl_H") else 0.0
            atoms.setdefault(sym, {})[(n, l)] = CSVLevel(E, xi)
            
    for sym, terms in sorted(invalid.items()):
        print(f"  [aviso] {sym}: filas no ligadas (E>=0) omitidas del CSV: {', '.join(terms)}")
    return atoms

def load_atoms_from_params(csv_path: str) -> dict[str, AlkaliAtom]:
    from collections import defaultdict
    params_by_sym = defaultdict(dict)
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sym = row["Elemento"]
            l = int(row["Canal_l"])
            params_by_sym[sym][l] = (
                float(row["a1"]), float(row["a2"]), float(row["a3"]),
                float(row["a4"]), float(row["r_c"])
            )
    atoms = {}
    for sym, p_dict in params_by_sym.items():
        if sym == "H":
            atoms[sym] = AlkaliAtom.make_hydrogenic(1)
            continue
        max_l = max(p_dict.keys())
        for l in range(6):
            if l not in p_dict: p_dict[l] = p_dict[max_l]
        try:
            sp = get_species(sym)
            atoms[sym] = AlkaliAtom(
                symbol=sp.symbol, Z=sp.Z, Z_c=sp.Z_c, n0=sp.n0, l0=sp.l0,
                is_tabulated=True, tabulated_params=p_dict, alpha_c=sp.alpha_c
            )
        except ValueError:
            pass
    return atoms

def reconstruct_radial_functions(atom: AlkaliAtom, csv_levels_sym: dict, n_points=50000):
    """
    Reconstruye P(r) buscando rapidísimo cerca de la E del CSV,
    y luego DESTRUYE el recálculo inyectándole la E y xi originales de vuelta.
    """
    n_max = max(n for (n, l) in csv_levels_sym.keys())
    r = build_radial_mesh(n_max=n_max, Z_c=atom.Z_c, n_points=n_points)
    radial = {}
    
    for (n, l), lvl in csv_levels_sym.items():
        V_func = lambda rr, ll=l: atom.V_eff(rr, ll)
        cand = find_bound_state_near(n, l, r, V_func, E_guess=lvl.E, window_frac=0.05, n_scan=10)
        
        cand.E = lvl.E
        cand.xi_nl = lvl.xi_nl
        cand.converged = True
        radial[(n, l)] = cand
        
    return radial, r

# =============================================================================
# Utilidades y Gráficas
# =============================================================================

def species_structure(radial_sols):
    n0 = min(n for (n, l) in radial_sols)
    l0 = min(l for (n, l) in radial_sols if n == n0)
    p_ns = [n for (n, l) in radial_sols if l == 1]
    n_p = min(p_ns) if p_ns else None
    return n0, l0, n_p

def atomic_to_tesla(B): return B * B_ATOMIC_TESLA
def crossover_field_atomic(xi_nl): return max(xi_nl, 1e-12) / zm.MU_B

def extract_lines(transitions, tol_nm=1e-3):
    lines = zm.merge_transition_lines(transitions, tol_nm=tol_nm)
    if not lines: return []
    imax = max(i for _, i in lines)
    return [(lam, i / imax) for lam, i in lines if i >= 1e-3 * imax]

def spectral_spread_nm(lines):
    lams = [lam for lam, _ in lines]
    return (max(lams) - min(lams)) if len(lams) >= 2 else 0.0

def figure_zeeman_map(states, n_p, outpath, sym):
    p_states = zm.filter_states(states, n=n_p, l=1)
    if not p_states: return 0.0
    
    E_p = p_states[0].E_nl
    xi_nl = p_states[0].xi_nl
    B_cross = crossover_field_atomic(xi_nl)
    B_arr = np.linspace(0.0, 5.0 * B_cross, 400)
    curves = zm.zeeman_field_scan(p_states, B_arr)
    B_T = atomic_to_tesla(B_arr)

    mj_vals = sorted(curves.keys())
    cmap = plt.get_cmap("turbo")
    span = (max(mj_vals) - min(mj_vals)) or 1.0

    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    seen = set()
    for mj in mj_vals:
        color = cmap(0.12 + 0.76 * (mj - min(mj_vals)) / span)
        E_rel = (curves[mj] - E_p) * HARTREE_TO_CM
        for k in range(E_rel.shape[1]):
            lbl = f"$m_j={mj:+.1f}$"
            ax.plot(B_T, E_rel[:, k], color=color, lw=1.7, label=lbl if lbl not in seen else None)
            seen.add(lbl)
    ax.axvline(atomic_to_tesla(B_cross), color="0.4", ls="--", lw=1.0)
    ax.text(atomic_to_tesla(B_cross), ax.get_ylim()[1], r"  $\mu_B B \approx \xi$", va="top", color="0.3", fontsize=9)
    ax.set_xlabel("Campo magnético $B_z$ [T]")
    ax.set_ylabel(rf"$E - E_{{{n_p}p}}$ [cm$^{{-1}}$]")
    ax.set_title(f"{sym}: desdoblamiento Zeeman del multiplete ${n_p}p$ ($\\xi$ = {xi_nl * HARTREE_TO_CM:.3g} cm$^{{-1}}$)")
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    return B_cross

def figure_emission_spectrum(lines0, linesB, B_demo, outpath, sym):
    B_T = atomic_to_tesla(B_demo)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8.0, 6.2), sharex=True)

    def stem(ax, lines, color, title):
        for lam, inten in lines:
            ax.vlines(lam, 0, inten, color=color, lw=2.0)
            ax.plot([lam], [inten], "o", color=color, ms=4)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel(r"$|D|^2$ relativo")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.2)

    stem(ax0, lines0, "#1f5fb4", f"{sym}: Espectro de emisión multicanal a $B=0$")
    stem(ax1, linesB, "#c0392b", f"{sym}: Espectro de emisión a $B={B_T:.3g}$ T (Efecto Zeeman)")
    ax1.set_xlabel(r"Longitud de onda $\lambda$ [nm]")
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)

# =============================================================================
# Franjas de Interferencia 
# =============================================================================

def figure_interference(lines0, linesB, B_demo, outpath, sym):
    lines0 = [(l, i) for l, i in lines0 if 100 < l < 15000]
    linesB = [(l, i) for l, i in linesB if 100 < l < 15000]
    
    if not lines0 and not linesB: return 0.0, 0
    all_lines = lines0 + linesB

    
    lam_dom = max(all_lines, key=lambda x: x[1])[0]

    
    L_nm = 1.0e9  
    Lam_target = 5.0e6  
    d_nm = lam_dom * L_nm / Lam_target  
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(8.4, 4.2), sharex=True)

    def screen(ax, lines, color, label):
        w_max = 0.25 * Lam_target  
        
        for lam, inten in lines:
            Lam_k = lam * L_nm / d_nm  
            for k in range(0, 11):
                yk = k * Lam_k
                half_w = 0.5 * w_max * inten 
                ax.axvspan((yk - half_w) / 1e6, (yk + half_w) / 1e6, color=color, alpha=0.85, lw=0)
                
        ax.set_yticks([])
        ax.set_ylabel(label, fontsize=10)
        ax.set_facecolor("0.08")

    B_T = atomic_to_tesla(B_demo)
    screen(axA, lines0, "#ffd95e", "$B = 0$")
    axA.set_title(f"{sym}: Franjas de doble rendija ($d\\sin\\theta=k\\lambda$)\n"
                  f"Ancho $\\propto |D|^2$  |  $d$ = {d_nm/1e6:.3g} mm, $L$ = 1 m", fontsize=11)
    
    screen(axB, linesB, "#ff6e5e", f"$B = {B_T:.3g}$ T")
    axB.set_xlabel("Posición en la pantalla desde el centro $y$ [mm]")
    axB.set_xlim(0, 45) 
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    return d_nm, 10


# =============================================================================
# Driver de Ejecución Rápida
# =============================================================================

def run_species(sym, atom, csv_levels_sym, outdir, current_idx, total_idx):

    print(f"\n[{current_idx}/{total_idx}] Procesando {sym} " + "="*40)

    tag = sym.replace("+", "plus")
    paths = {
        k: os.path.join(outdir, f"{tag}_plot_{k}.png")
        for k in ("zeeman_map", "emission_spectrum", "interference_fringes")
    }

    # Reconstrucción radial
    print("  -> 1/5: Reconstruyendo funciones radiales desde el CSV...",
          end="", flush=True)
    t_step = time.time()

    radial_sols, r_mesh = reconstruct_radial_functions(
        atom,
        csv_levels_sym,
        n_points=50000
    )

    print(f" Listo en {time.time()-t_step:.1f}s")

    n0, l0, n_p = species_structure(radial_sols)

    if n_p is None:
        print("  [!] sin canal p — omitido")
        return None

    states = zm.build_subspace_states(
        n0,
        l0,
        radial_sols,
        extra_n=2
    )

    #Mapa Zeeman
     print("  -> 2/5: Resolviendo mapa Zeeman del multiplete p...",
          end="", flush=True)
    t_step = time.time()

    B_cross = figure_zeeman_map(
        states,
        n_p,
        paths["zeeman_map"],
        sym
    )

    B_demo = 0.6 * B_cross

    print(f" Listo en {time.time()-t_step:.1f}s")

    # Diagonalización
    
    print("  -> 3/5: Diagonalizando subespacios completos (B=0 y B>0)...",
          end="", flush=True)
    t_step = time.time()

    blocks0 = zm.diagonalize_subspace(states, 0.0)
    blocksB = zm.diagonalize_subspace(states, B_demo)

    print(f" Listo en {time.time()-t_step:.1f}s")

    # Espectro de emisión
    
    print("  -> 4/5: Calculando matriz de dipolos y símbolos 3j de Wigner...",
          end="", flush=True)
    t_step = time.time()

    trans0 = zm.emission_spectrum(blocks0, radial_sols, r_mesh)
    transB = zm.emission_spectrum(blocksB, radial_sols, r_mesh)

    lines0 = extract_lines(trans0)
    linesB = extract_lines(transB)

    print(f" Listo en {time.time()-t_step:.1f}s")

    if not lines0:
        print(f"  [!] {sym}: sin transiciones permitidas generadas — omitido")
        return None

    # Franjas e imágenes
    print("  -> 5/5: Evaluando sumatorias para franjas de interferencia...",
          end="", flush=True)
    t_step = time.time()

    figure_emission_spectrum(
        lines0,
        linesB,
        B_demo,
        paths["emission_spectrum"],
        sym
    )

    d_nm, k_star = figure_interference(
        lines0,
        linesB,
        B_demo,
        paths["interference_fringes"],
        sym
    )

    print(f" Listo en {time.time()-t_step:.1f}s")

    print(
        f"  [OK] {sym} finalizado. "
        f"B_cross={atomic_to_tesla(B_cross):.3g} T, "
        f"líneas B=0/B>0: {len(lines0)}/{len(linesB)}"
    )

    return paths


def main():
    ap = argparse.ArgumentParser(description="Zeeman multicanal con R real y E, xi del CSV")
    ap.add_argument("--csv", default="out/alkali_levels.csv", help="Tu CSV original")
    ap.add_argument("--params", default="Parametros.csv", help="Tu CSV de parámetros")
    ap.add_argument("--species", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--outdir", default="out_zeeman")
    args = ap.parse_args()

    if not os.path.exists(args.csv) or not os.path.exists(args.params):
        print(f"[!] Faltan archivos. Verifica {args.csv} y {args.params}")
        return

    csv_data = load_levels_csv(args.csv)
    atoms_data = load_atoms_from_params(args.params)
    
    if args.all: targets = sorted(csv_data.keys())
    elif args.species: targets = [s.strip() for s in args.species.split(",")]
    else: targets = ["Na"]

    os.makedirs(args.outdir, exist_ok=True)
    print(f"\n[INICIO] Usando {args.csv} (E, xi) y {args.params} (R)")
    
    for idx, sym in enumerate(targets, 1):
        if sym in csv_data and sym in atoms_data:
            run_species(sym, atoms_data[sym], csv_data[sym], args.outdir, idx, len(targets))
            
    print(f"\n[FIN] Todo guardado en '{args.outdir}/'")

if __name__ == "__main__":
    main()
