"""
zeeman_matrix.py — efecto Zeeman sin diamagnetismo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np
from scipy.integrate import simpson

from alkali_radial_solver import (
    ALPHA_FS,
    SPEED_LIGHT,
    HARTREE_TO_CM,
    build_radial_mesh,
    find_bound_state,
    find_bound_state_near,
    compute_xi_nl,
    RadialSolution,
)

MU_B = 0.5
H_PLANCK = 2.0 * math.pi
HC_ATOMIC = H_PLANCK * SPEED_LIGHT
BOHR_TO_NM = 0.052917721090
B_ATOMIC_TESLA = 235051.756


@dataclass(frozen=True)
class BasisState:
    n: int
    l: int
    j: float
    mj: float
    E_nl: float
    xi_nl: float


@dataclass
class BlockSolution:
    mj: float
    states: list[BasisState]
    energies: np.ndarray
    vectors: np.ndarray


@dataclass
class Transition:
    E_i: float
    E_f: float
    dE: float
    wavelength_nm: float
    intensity: float


def lande_g(l: int, j: float) -> float:
    return 1.0 + (j * (j + 1.0) + 0.75 - l * (l + 1.0)) / (2.0 * j * (j + 1.0))


def spin_orbit_factor(l: int, j: float) -> float:
    return 0.5 * (j * (j + 1.0) - l * (l + 1.0) - 0.75)


def clebsch_gordan_spinor(l: int, j: float, mj: float) -> tuple[float, float]:
    denom = 2.0 * l + 1.0
    if abs(j - (l + 0.5)) < 1e-9:
        c_plus = math.sqrt((l + mj + 0.5) / denom)
        c_minus = math.sqrt((l - mj + 0.5) / denom)
    else:
        c_plus = -math.sqrt((l - mj + 0.5) / denom)
        c_minus = math.sqrt((l + mj + 0.5) / denom)
    return c_plus, c_minus


@lru_cache(maxsize=None)
def _wigner_3j(j1: float, j2: float, j3: float,
               m1: float, m2: float, m3: float) -> float:
    from sympy import Rational
    from sympy.physics.wigner import wigner_3j as _w3j

    def rat(x: float):
        return Rational(round(2 * x), 2)

    return float(_w3j(rat(j1), rat(j2), rat(j3), rat(m1), rat(m2), rat(m3)))


def gaunt_integral(lf: int, mlf: float, li: int, mli: float, u: int) -> float:
    if int(round(mli)) != int(round(mlf - u)):
        return 0.0
    pref = ((-1.0) ** int(round(mlf))) * math.sqrt((2 * lf + 1) * (2 * li + 1))
    w0 = _wigner_3j(lf, 1, li, 0, 0, 0)
    if w0 == 0.0:
        return 0.0
    wm = _wigner_3j(lf, 1, li, -mlf, u, mli)
    return pref * w0 * wm


def radial_dipole_integral(P_f: np.ndarray, P_i: np.ndarray, r: np.ndarray) -> float:
    return float(simpson(P_f * r * P_i, r))


def build_subspace_states(n0: int, l0: int,
                          radial: dict[tuple[int, int], RadialSolution],
                          extra_n: int = 1) -> list[BasisState]:
    states: list[BasisState] = []
    n_values = list(range(n0, n0 + 1 + extra_n))
    for n in n_values:
        for l in range(0, n):
            if n == n0 and l < l0:
                continue
            sol = radial.get((n, l))
            if sol is None or not sol.converged:
                continue
            j_values = [0.5] if l == 0 else [l + 0.5, l - 0.5]
            for j in j_values:
                mj = -j
                while mj <= j + 1e-9:
                    states.append(BasisState(n=n, l=l, j=j, mj=round(mj, 1),
                                             E_nl=sol.E, xi_nl=sol.xi_nl))
                    mj += 1.0
    return states


def group_by_mj(states: list[BasisState]) -> dict[float, list[BasisState]]:
    blocks: dict[float, list[BasisState]] = {}
    for s in states:
        blocks.setdefault(s.mj, []).append(s)
    return blocks


def filter_states(states: list[BasisState],
                  n: int | None = None,
                  l: int | None = None) -> list[BasisState]:
    out = states
    if n is not None:
        out = [s for s in out if s.n == n]
    if l is not None:
        out = [s for s in out if s.l == l]
    return out


def build_block_Hmj(block_states: list[BasisState], Bz: float) -> np.ndarray:
    D = len(block_states)
    H = np.zeros((D, D), dtype=float)
    for i, si in enumerate(block_states):
        gL = lande_g(si.l, si.j)
        H[i, i] = (si.E_nl
                   + gL * si.mj * MU_B * Bz
                   + si.xi_nl * spin_orbit_factor(si.l, si.j))
        for k in range(i + 1, D):
            sk = block_states[k]
            if sk.n == si.n and sk.l == si.l and abs(sk.j - si.j) == 1.0:
                l = si.l
                off = -MU_B * Bz * math.sqrt((l + 0.5) ** 2 - si.mj ** 2) / (2 * l + 1)
                H[i, k] = off
                H[k, i] = off
    return H


def diagonalize_subspace(states: list[BasisState], Bz: float) -> list[BlockSolution]:
    blocks = group_by_mj(states)
    out: list[BlockSolution] = []
    for mj in sorted(blocks.keys()):
        bstates = blocks[mj]
        H = build_block_Hmj(bstates, Bz)
        evals, evecs = np.linalg.eigh(H)
        out.append(BlockSolution(mj=mj, states=bstates, energies=evals, vectors=evecs))
    return out


def zeeman_field_scan(states: list[BasisState],
                      B_array: np.ndarray) -> dict[float, np.ndarray]:
    blocks = group_by_mj(states)
    result: dict[float, np.ndarray] = {}
    for mj, bstates in blocks.items():
        D = len(bstates)
        curve = np.zeros((len(B_array), D))
        prev = None
        for ib, B in enumerate(B_array):
            H = build_block_Hmj(bstates, float(B))
            evals = np.linalg.eigvalsh(H)
            if prev is None:
                order = np.argsort(evals)
                curve[ib] = evals[order]
            else:
                ev_sorted = np.sort(evals)
                used = [False] * D
                assigned = np.empty(D)
                for p in range(D):
                    dists = [abs(ev_sorted[q] - prev[p]) if not used[q] else math.inf
                             for q in range(D)]
                    q = int(np.argmin(dists))
                    used[q] = True
                    assigned[p] = ev_sorted[q]
                curve[ib] = assigned
            prev = curve[ib]
        result[mj] = curve
    return result


def _basis_dipole_amplitude(sf: BasisState, si: BasisState,
                            R: float, q: int) -> complex:
    cfp, cfm = clebsch_gordan_spinor(sf.l, sf.j, sf.mj)
    cip, cim = clebsch_gordan_spinor(si.l, si.j, si.mj)
    g_up = gaunt_integral(sf.l, sf.mj - 0.5, si.l, si.mj - 0.5, q)
    g_dn = gaunt_integral(sf.l, sf.mj + 0.5, si.l, si.mj + 0.5, q)
    return R * (cfp * cip * g_up + cfm * cim * g_dn)


def emission_spectrum(block_solutions: list[BlockSolution],
                      radial: dict[tuple[int, int], RadialSolution],
                      r_common: np.ndarray,
                      intensity_tol: float = 1e-12) -> list[Transition]:
    R_cache: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}

    def R_of(nl_f: tuple[int, int], nl_i: tuple[int, int]) -> float:
        key = (nl_f, nl_i)
        if key not in R_cache:
            sf = radial[nl_f]
            si = radial[nl_i]
            R_cache[key] = radial_dipole_integral(sf.P, si.P, r_common)
        return R_cache[key]

    eig_states = []
    for ibk, bk in enumerate(block_solutions):
        for ie in range(len(bk.energies)):
            eig_states.append((bk.energies[ie], bk.mj, ibk, ie))

    transitions: list[Transition] = []
    n_eig = len(eig_states)
    for a in range(n_eig):
        Ea, mja, bka, iea = eig_states[a]
        for b in range(n_eig):
            if a == b:
                continue
            Eb, mjb, bkb, ieb = eig_states[b]
            if not (Eb < Ea):
                continue
            E_i, E_f = Ea, Eb
            q = int(round(mja - mjb))
            if q not in (-1, 0, 1):
                continue

            blk_i = block_solutions[bka]
            blk_f = block_solutions[bkb]
            c_i = blk_i.vectors[:, iea]
            c_f = blk_f.vectors[:, ieb]

            D_q = 0.0 + 0.0j
            for kf, sf in enumerate(blk_f.states):
                cf = c_f[kf]
                if cf == 0.0:
                    continue
                for ki, sii in enumerate(blk_i.states):
                    ci = c_i[ki]
                    if ci == 0.0:
                        continue
                    # Eliminada la condición de Δℓ = ±1.
                    R = R_of((sf.n, sf.l), (sii.n, sii.l))
                    if R == 0.0:
                        continue
                    amp = _basis_dipole_amplitude(sf, sii, R, q)
                    D_q += np.conj(cf) * ci * amp

            intensity = float(abs(D_q) ** 2)
            if intensity <= intensity_tol:
                continue

            dE = E_i - E_f
            inv_lambda = dE / HC_ATOMIC
            lam_bohr = 1.0 / inv_lambda if inv_lambda > 0 else float("inf")
            lam_nm = lam_bohr * BOHR_TO_NM
            transitions.append(Transition(E_i=E_i, E_f=E_f, dE=dE,
                                          wavelength_nm=lam_nm,
                                          intensity=intensity))
    return transitions


def merge_transition_lines(transitions: list[Transition],
                           tol_nm: float = 1e-4) -> list[tuple[float, float]]:
    if not transitions:
        return []
    items = sorted(transitions, key=lambda t: t.wavelength_nm)
    merged: list[list[float]] = []
    for t in items:
        if merged and abs(t.wavelength_nm - merged[-1][0]) < tol_nm:
            merged[-1][1] += t.intensity
        else:
            merged.append([t.wavelength_nm, t.intensity])
    return [(lam, inten) for lam, inten in merged]


def interference_pattern(lines: list[tuple[float, float]],
                         d: float, L: float,
                         y: np.ndarray) -> np.ndarray:
    I = np.zeros_like(y, dtype=float)
    for lam, inten in lines:
        if lam <= 0 or not math.isfinite(lam):
            continue
        I += inten * np.cos(math.pi * d * y / (lam * L)) ** 2
    return I


def solve_subspace_common_mesh(atom, extra_n: int = 1,
                               n_points: int = 200000,
                               verbose: bool = False
                               ) -> tuple[dict[tuple[int, int], RadialSolution], np.ndarray]:
    n0 = atom.n0
    l0 = atom.l0
    n_max = n0 + extra_n
    r = build_radial_mesh(n_max=n_max, Z_c=atom.Z_c, n_points=n_points)

    radial: dict[tuple[int, int], RadialSolution] = {}
    for n in range(n0, n_max + 1):
        for l in range(0, n):
            if n == n0 and l < l0:
                continue
            V_func = lambda rr, ll=l: atom.V_eff(rr, ll)
            sol = None
            if (n, l) in atom.nist_levels:
                E_guess = atom.nist_levels[(n, l)] - atom.E_ionization_H
                try:
                    cand = find_bound_state_near(n, l, r, V_func, E_guess=E_guess,
                                                 window_frac=0.4, n_scan=25)
                    if cand.converged and abs(cand.E - E_guess) < 0.5 * abs(E_guess):
                        sol = cand
                except Exception:
                    sol = None
            if sol is None:
                sol = find_bound_state(n, l, r, V_func, Z_c=atom.Z_c)
            if sol.converged:
                sol.xi_nl = compute_xi_nl(sol, lambda rr, ll=l: atom.dV_eff(rr, ll))
            radial[(n, l)] = sol
            if verbose:
                tag = "ok" if sol.converged else "NO CONV"
                print(f"    ({n},{l}) E={sol.E:.6f} xi={sol.xi_nl:.3e} [{tag}]")
    return radial, r
