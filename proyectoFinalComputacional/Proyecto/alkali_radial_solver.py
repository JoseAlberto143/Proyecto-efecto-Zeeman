"""
=============================================================================
SOLUCIONADOR RADIAL PARA ÁTOMOS TIPO ALCALINO
=============================================================================

Este módulo resuelve la ecuación radial reducida de Schrödinger para el
electrón de valencia de un átomo tipo alcalino, usando:

    * Pseudopotencial de Marinescu-Dalgarno Generalizado 
      (PRA 49, 982, 1994 - Ec. 18a y 18b adaptada con Z_c para iones):

          V_eff(r) = -(1/r) * [Z_c + (Z-Z_c)*exp(-a1*r) - r*(a3 + a4*r)*exp(-a2*r)]
                     - (alpha_c / (2*r^4)) * [1 - exp(-(r/r_c)^6)]

      Z_c = q + 1 (carga residual del core)
      Cinco parámetros a ajustar: a1, a2, a3, a4, r_c.
      alpha_c se fija externamente.

    * Método de Numerov para integrar la ecuación radial reducida:

          d^2 P / dr^2 = [ ℓ(ℓ+1)/r^2 + 2·V_eff(r) - 2·E ] · P(r)

    * Shooting bidireccional con criterio de empalme por Wronskiano.

    * Teoría de Defecto Cuántico (QDT) para extraer energías de referencia
      a partir de los datos del NIST y ajustar los parámetros.

Unidades atómicas en todo el módulo: ħ = m_e = e = 4πε_0 = 1.

=============================================================================
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.integrate import simpson
from scipy.optimize import brentq, minimize


from numerov_fast import shoot_wronskian_fast, warmup_jit, NUMBA_AVAILABLE


print(f"{NUMBA_AVAILABLE}")

# =============================================================================
# CONSTANTES FÍSICAS Y CONVERSIONES
# =============================================================================

ALPHA_FS = 1.0 / 137.035999084          # constante de estructura fina
SPEED_LIGHT = 1.0 / ALPHA_FS            # velocidad de la luz en unidades atómicas
HARTREE_TO_EV = 27.211386245988         # 1 Hartree = 27.2114 eV
HARTREE_TO_CM = 219474.63136320         # 1 Hartree = 219474.63 cm^-1
BOHR_TO_ANGSTROM = 0.529177210903


# =============================================================================
# DATOS TABULADOS Y CONFIGURACIONES
# =============================================================================

# Configuraciones electrónicas 
ALKALI_GROUND_STATES: dict[str, dict] = {
    "H":  {"Z": 1,  "n0": 1, "l0": 0, "E_ion_H": 0.499735},  
    "Li": {"Z": 3,  "n0": 2, "l0": 0, "E_ion_H": 0.198142},
    "Na": {"Z": 11, "n0": 3, "l0": 0, "E_ion_H": 0.188858},
    "K":  {"Z": 19, "n0": 4, "l0": 0, "E_ion_H": 0.159517},
    "Rb": {"Z": 37, "n0": 5, "l0": 0, "E_ion_H": 0.153507},
    "Cs": {"Z": 55, "n0": 6, "l0": 0, "E_ion_H": 0.143093},
    "Fr": {"Z": 87, "n0": 7, "l0": 0, "E_ion_H": 0.149667},
}

# Parámetros de Marinescu-Dalgarno.
MARINESCU_PARAMS: dict[str, dict] = {
    "Li": {
        "alpha_c": 0.1923,
        0: (2.47718079, 1.84150932, -0.02169712, -0.11988362, 0.61340824),
        1: (3.45414648, 2.55151080, -0.21646561, -0.06990078, 0.61566441),
        2: (2.51909839, 2.43712450,  0.32505524,  0.10602430, 2.34126273),
        3: (2.51909839, 2.43712450,  0.32505524,  0.10602430, 2.34126273),
    },
    "Na": {
        "alpha_c": 0.9448,
        0: (4.82223117, 2.45449865, -1.12255048, -1.42631393, 0.45489422),
        1: (5.08382502, 2.18228758, -1.19534623, -1.03142861, 0.45798739),
        2: (3.53324124, 2.48697936, -0.75688448, -1.27852357, 0.71875312),
        3: (1.11056646, 1.05458759,  1.73203428, -0.09265696, 28.67350509),
    },
    "K": {
        "alpha_c": 5.3310,
        0: (3.56079437, 1.83909642, -1.74701102, -1.03237313, 0.83167545),
        1: (3.65670429, 1.67520788, -2.07416615, -0.89030421, 0.85235381),
        2: (4.12713694, 1.79837462, -1.69935171, -0.98913582, 0.83216907),
        3: (1.42310446, 1.27861156,  4.77441476, -0.94829262, 6.50294371),
    },
    "Rb": {
        "alpha_c": 9.0760,
        0: (3.69628474, 1.64915255, -9.86069196,  0.19579987, 1.66242117),
        1: (4.44088978, 1.92828831, -16.79597770, -0.81633314, 1.50195124),
        2: (3.78717363, 1.57027864, -11.65588970, 0.52942835, 4.86851938),
        3: (2.39848933, 1.76810544, -12.07106780, 0.77256589, 4.79831327),
    },
    "Cs": {
        "alpha_c": 15.6440,
        0: (3.49546309, 1.47533800, -9.72143084,  0.02629242, 1.92046930),
        1: (4.69366096, 1.71398344, -24.65624280, -0.09543125, 2.13383095),
        2: (4.32466196, 1.61365288, -6.70128850, -0.74095193, 0.93007296),
        3: (3.01048361, 1.40000001, -3.20036138, 0.00034538, 1.99969677),
    },
}

NIST_REFERENCE_LEVELS: dict[str, dict] = {
    "Li": {
        (2, 0): 0.000000, (2, 1): 0.067907, (3, 0): 0.123969, (3, 1): 0.140914,
        (3, 2): 0.142544, (4, 0): 0.159528, (4, 1): 0.166168, (4, 2): 0.166869,
        (4, 3): 0.166900, (5, 0): 0.172941, (5, 1): 0.175760,
    },
    "Na": {
        (3, 0): 0.000000, (3, 1): 0.077310, (3, 2): 0.132924, (4, 0): 0.117279,
        (4, 1): 0.137906, (4, 2): 0.157411, (4, 3): 0.157593, (5, 0): 0.150289,
        (5, 1): 0.159419,
    },
    "K": {
        (4, 0): 0.000000, (4, 1): 0.059340, (3, 2): 0.098121, (5, 0): 0.095804,
        (5, 1): 0.112634, (4, 2): 0.125075, (4, 3): 0.124834, (6, 0): 0.118987,
        (6, 1): 0.127340,
    },
    "Rb": {
        (5, 0): 0.000000, (5, 1): 0.058036, (4, 2): 0.088187, (6, 0): 0.091728,
        (6, 1): 0.108055, (5, 2): 0.118987, (4, 3): 0.115170, (7, 0): 0.115170,
    },
    "Cs": {
        (6, 0): 0.000000, (6, 1): 0.052614, (5, 2): 0.066068, (7, 0): 0.084455,
        (7, 1): 0.099170, (6, 2): 0.108061, (4, 3): 0.099996, (8, 0): 0.106650,
    },
}

# =============================================================================
# CLASES DE DATOS
# =============================================================================

@dataclass
class PseudoPotentialParams:
    """Parámetros del pseudopotencial Marinescu-Dalgarno generalizado."""
    Z: int           
    Z_c: int         
    a1: float
    a2: float
    a3: float
    a4: float        
    alpha_c: float   
    r_c: float       
    l: int = 0       


@dataclass
class RadialSolution:
    n: int
    l: int
    E: float                 
    r: np.ndarray            
    P: np.ndarray            
    converged: bool
    n_nodes: int             
    xi_nl: float = 0.0       


# =============================================================================
# PSEUDOPOTENCIAL MARINESCU-DALGARNO GENERALIZADO
# =============================================================================

def V_eff_marinescu_gen(r: np.ndarray, p: PseudoPotentialParams) -> np.ndarray:
    """Pseudopotencial efectivo universal para átomos tipo alcalino."""
    r_safe = np.where(r > 1e-12, r, 1e-12)
    
    Z_l = (
        p.Z_c 
        + (p.Z - p.Z_c) * np.exp(-p.a1 * r_safe) 
        - r_safe * (p.a3 + p.a4 * r_safe) * np.exp(-p.a2 * r_safe)
    )
    V_central = -Z_l / r_safe
    V_pol = -p.alpha_c / (2.0 * r_safe**4) * (1.0 - np.exp(-((r_safe / p.r_c)**6)))
    
    return V_central + V_pol


def dV_eff_marinescu_gen(r: np.ndarray, p: PseudoPotentialParams) -> np.ndarray:
    """Derivada analítica exacta de V_eff_marinescu_gen para espín-órbita."""
    r_safe = np.where(r > 1e-12, r, 1e-12)
    
    term1 = p.Z_c / (r_safe**2)
    term2 = (p.Z - p.Z_c) * np.exp(-p.a1 * r_safe) / (r_safe**2)
    term3 = p.a1 * (p.Z - p.Z_c) * np.exp(-p.a1 * r_safe) / r_safe
    term4 = p.a4 * np.exp(-p.a2 * r_safe)
    term5 = -p.a2 * (p.a3 + p.a4 * r_safe) * np.exp(-p.a2 * r_safe)
    dV_central = term1 + term2 + term3 + term4 + term5
    
    exp_factor = np.exp(-((r_safe / p.r_c)**6))
    dV_pol = (2.0 * p.alpha_c / r_safe**5) * (1.0 - exp_factor) + (3.0 * p.alpha_c * r_safe / p.r_c**6) * exp_factor
    
    return dV_central + dV_pol


# =============================================================================
# MALLA RADIAL 
# =============================================================================

def build_radial_mesh(n_max: int, Z_c: int, n_points: int = 200000, r_min: float = 1e-5) -> np.ndarray:
    """Malla radial lineal optimizada para resolver el estado |n_max, l>.

    El extremo r_max se elige de modo que P_in ~ exp(-kappa·r_max) sea negligible
    (~1e-13 para los estados ligados del subespacio) sin desperdiciar densidad
    de puntos en la región asintótica. Para Z_c=1 (alcalinos neutros) y
    n_max≲8 se obtiene h ~ 0.02 Bohr cerca del origen, suficiente para
    resolver el core del pseudopotencial Marinescu-Dalgarno sobre una malla
    100% lineal sin recurrir a esquemas log-lineales híbridos.

    La fórmula es r_max ≈ 30·n_max/Z_c, acotada inferiormente a 40 Bohr
    y superiormente a 250 Bohr .
    """
    Zc = max(Z_c, 1)
    r_max = 30.0 * float(n_max) / Zc
    r_max = max(40.0, min(r_max, 250.0))
    return np.linspace(r_min, r_max, n_points)


def numerov_outward(r: np.ndarray, V: np.ndarray, E: float, l: int, m_match: int) -> np.ndarray:
    N = len(r)
    P = np.zeros(N)
    h = r[1] - r[0]
    r_safe = np.where(r > 1e-12, r, 1e-12)
    f = l * (l + 1) / r_safe**2 + 2.0 * V - 2.0 * E

    P[0] = 0.0
    P[1] = h ** (l + 1) if l >= 0 else h
    h2_12 = h * h / 12.0
    
    for i in range(1, m_match):
        c_curr = 2.0 * (1.0 + 5.0 * h2_12 * f[i]) * P[i]
        c_prev = (1.0 - h2_12 * f[i - 1]) * P[i - 1]
        denom = 1.0 - h2_12 * f[i + 1]
        if abs(denom) < 1e-14:
            denom = 1e-14
        P[i + 1] = (c_curr - c_prev) / denom
        if abs(P[i + 1]) > 1e20:
            scale = 1e20 / abs(P[i + 1])
            P[: i + 2] *= scale
    return P


def numerov_inward(r: np.ndarray, V: np.ndarray, E: float, l: int, m_match: int) -> np.ndarray:
    N = len(r)
    P = np.zeros(N)
    h = r[1] - r[0]
    r_safe = np.where(r > 1e-12, r, 1e-12)
    f = l * (l + 1) / r_safe**2 + 2.0 * V - 2.0 * E

    kappa = math.sqrt(-2.0 * E) if E < 0 else 1.0
    P[N - 1] = math.exp(-kappa * r[N - 1])
    P[N - 2] = math.exp(-kappa * r[N - 2])

    if P[N - 1] < 1e-50:
        P[N - 1] = 1e-50
        P[N - 2] = 1e-50 * math.exp(kappa * h)

    h2_12 = h * h / 12.0
    for i in range(N - 2, m_match, -1):
        c_curr = 2.0 * (1.0 + 5.0 * h2_12 * f[i]) * P[i]
        c_next = (1.0 - h2_12 * f[i + 1]) * P[i + 1]
        denom = 1.0 - h2_12 * f[i - 1]
        if abs(denom) < 1e-14:
            denom = 1e-14
        P[i - 1] = (c_curr - c_next) / denom
        if abs(P[i - 1]) > 1e20:
            scale = 1e20 / abs(P[i - 1])
            P[i - 1 :] *= scale
    return P


def find_classical_turning_point(r: np.ndarray, V: np.ndarray, E: float, l: int) -> int:
    r_safe = np.where(r > 1e-12, r, 1e-12)
    V_eff_total = V + l * (l + 1) / (2.0 * r_safe**2) 
    kinetic = E - V_eff_total
    pos = np.where(kinetic > 0)[0]
    if len(pos) == 0:
        return len(r) // 2
    return int(pos[-1])


def wronskian_at_match(P_out: np.ndarray, P_in: np.ndarray, m_match: int) -> float:
    if abs(P_in[m_match]) < 1e-300:
        return float("inf")
    scale = P_out[m_match] / P_in[m_match]
    P_in_scaled = P_in * scale
    return (P_in_scaled[m_match + 1] - P_out[m_match + 1] + P_out[m_match - 1] - P_in_scaled[m_match - 1])


def shoot_wronskian(E: float, r: np.ndarray, V: np.ndarray, l: int) -> tuple[float, np.ndarray, int]:
    W, P_combined, m_match = shoot_wronskian_fast(E, r, V, l)
    return float(W), P_combined, int(m_match)


def shoot_wronskian_pure_python(E: float, r: np.ndarray, V: np.ndarray, l: int) -> tuple[float, np.ndarray, int]:
    m_match = find_classical_turning_point(r, V, E, l)
    m_match = max(5, min(len(r) - 5, m_match))

    P_out = numerov_outward(r, V, E, l, m_match + 2)
    P_in = numerov_inward(r, V, E, l, m_match - 2)
    W = wronskian_at_match(P_out, P_in, m_match)

    if abs(P_in[m_match]) > 1e-300:
        scale = P_out[m_match] / P_in[m_match]
        P_combined = np.copy(P_out)
        P_combined[m_match:] = P_in[m_match:] * scale
    else:
        P_combined = np.copy(P_out)
    return W, P_combined, m_match


def normalize_radial(P: np.ndarray, r: np.ndarray) -> np.ndarray:
    norm2 = simpson(P**2, r)
    if norm2 <= 0:
        return P
    return P / math.sqrt(norm2)


def count_nodes(P: np.ndarray, tol: float = 1e-8) -> int:
    abs_max = np.max(np.abs(P))
    if abs_max == 0:
        return 0
    P_clean = np.where(np.abs(P) > tol * abs_max, P, 0.0)
    nonzero = P_clean[P_clean != 0]
    if nonzero.size < 2:
        return 0
    signs = np.sign(nonzero)
    return int(np.sum(np.diff(signs) != 0))


# =============================================================================
# BÚSQUEDA DE ENERGÍAS LIGADAS
# =============================================================================

def find_bound_state(
    n: int, l: int, r: np.ndarray, V_func: Callable[[np.ndarray], np.ndarray],
    E_min: float | None = None, E_max: float | None = None, Z_c: int = 1, n_scan: int = 250,
) -> RadialSolution:
    V = V_func(r)
    if E_min is None:
        E_min = -Z_c**2 / 2.0 * 1.05 
    if E_max is None:
        E_max = -1e-5

    energies = np.linspace(E_min, E_max, n_scan)
    W_values = []
    target_nodes = n - l - 1

    for E in energies:
        try:
            W, _, _ = shoot_wronskian(E, r, V, l)
        except Exception:
            W = float("nan")
        W_values.append(W)

    W_values = np.array(W_values)
    roots: list[tuple[float, float]] = []
    
    for i in range(len(energies) - 1):
        if not (math.isfinite(W_values[i]) and math.isfinite(W_values[i + 1])):
            continue
        if W_values[i] * W_values[i + 1] < 0:
            roots.append((energies[i], energies[i + 1]))

    if not roots:
        warnings.warn(f"No se encontraron raíces del Wronskiano para n={n}, l={l}.")
        return RadialSolution(n=n, l=l, E=float("nan"), r=r, P=np.zeros_like(r), converged=False, n_nodes=-1)

    for E_a, E_b in roots:
        try:
            E_root = brentq(lambda EE: shoot_wronskian(EE, r, V, l)[0], E_a, E_b, xtol=1e-10, maxiter=120)
        except Exception:
            continue
        _, P_combined, _ = shoot_wronskian(E_root, r, V, l)
        P_norm = normalize_radial(P_combined, r)
        n_nodes_found = count_nodes(P_norm)

        if n_nodes_found == target_nodes:
            return RadialSolution(n=n, l=l, E=E_root, r=r, P=P_norm, converged=True, n_nodes=n_nodes_found)

    E_a, E_b = roots[0]
    try:
        E_root = brentq(lambda EE: shoot_wronskian(EE, r, V, l)[0], E_a, E_b, xtol=1e-10, maxiter=120)
        _, P_combined, _ = shoot_wronskian(E_root, r, V, l)
        P_norm = normalize_radial(P_combined, r)
        warnings.warn(f"Aviso (n={n}, l={l}): {count_nodes(P_norm)} nodos (esperado {target_nodes}).")
        return RadialSolution(n=n, l=l, E=E_root, r=r, P=P_norm, converged=True, n_nodes=count_nodes(P_norm))
    except Exception:
        return RadialSolution(n=n, l=l, E=float("nan"), r=r, P=np.zeros_like(r), converged=False, n_nodes=-1)


def find_bound_state_near(
    n: int, l: int, r: np.ndarray, V_func: Callable[[np.ndarray], np.ndarray],
    E_guess: float, window_frac: float = 0.35, n_scan: int = 30,
) -> RadialSolution:
    V = V_func(r)
    delta = abs(E_guess) * window_frac
    E_min = E_guess - delta
    E_max = min(E_guess + delta, -1e-6)
    if E_min >= E_max:
        E_min = E_guess * 2.0 

    energies = np.linspace(E_min, E_max, n_scan)
    W_values = []
    
    for E in energies:
        try:
            W, _, _ = shoot_wronskian(E, r, V, l)
        except Exception:
            W = float("nan")
        W_values.append(W)
        
    W_values = np.array(W_values)
    best_root: tuple[float, float] | None = None
    best_dist = float("inf")
    
    for i in range(len(energies) - 1):
        if not (math.isfinite(W_values[i]) and math.isfinite(W_values[i + 1])):
            continue
        if W_values[i] * W_values[i + 1] < 0:
            E_mid = 0.5 * (energies[i] + energies[i + 1])
            d = abs(E_mid - E_guess)
            if d < best_dist:
                best_dist = d
                best_root = (energies[i], energies[i + 1])

    if best_root is None:
        return find_bound_state(n, l, r, V_func, n_scan=120)

    try:
        E_root = brentq(lambda EE: shoot_wronskian(EE, r, V, l)[0], best_root[0], best_root[1], xtol=1e-9, maxiter=80)
        _, P_combined, _ = shoot_wronskian(E_root, r, V, l)
        P_norm = normalize_radial(P_combined, r)
        return RadialSolution(n=n, l=l, E=E_root, r=r, P=P_norm, converged=True, n_nodes=count_nodes(P_norm))
    except Exception:
        return RadialSolution(n=n, l=l, E=float("nan"), r=r, P=np.zeros_like(r), converged=False, n_nodes=-1)


def compute_xi_nl(sol: RadialSolution, dV_func: Callable[[np.ndarray], np.ndarray]) -> float:
    if not sol.converged:
        return 0.0
    r_safe = np.where(sol.r > 1e-12, sol.r, 1e-12)
    integrand = (ALPHA_FS**2 / (2.0 * r_safe)) * sol.P**2 * dV_func(sol.r)
    return float(simpson(integrand, sol.r))


# =============================================================================
# TEORÍA DE DEFECTO CUÁNTICO
# =============================================================================

def quantum_defect_from_energy(n: int, E_nl_Hartree: float, Z_c: int) -> float:
    if E_nl_Hartree >= 0:
        return float("nan")
    return n - Z_c / math.sqrt(-2.0 * E_nl_Hartree)


def qdt_average_defects(levels_relative: dict[tuple[int, int], float], E_ionization_H: float, Z_c: int) -> dict[int, float]:
    levels_absolute = {nl: E_rel - E_ionization_H for nl, E_rel in levels_relative.items()}
    by_l: dict[int, list[float]] = {}
    for (n, l), E in levels_absolute.items():
        if E >= 0:
            continue
        delta = quantum_defect_from_energy(n, E, Z_c)
        if math.isfinite(delta):
            by_l.setdefault(l, []).append(delta)
    return {l: float(np.mean(deltas)) for l, deltas in by_l.items()}


def qdt_reference_energy(n: int, delta_l: float, Z_c: int) -> float:
    return -(Z_c**2) / (2.0 * (n - delta_l) ** 2)


def fit_pseudopotential_params(
    Z: int,
    Z_c: int,
    levels_relative: dict[tuple[int, int], float],
    E_ionization_H: float,
    alpha_c_fixed: float = 0.0,
    initial_guess: tuple[float, ...] | None = None,
    n_pairs_max: int = 6,
    max_iter: int = 150,
    verbose: bool = False,
) -> tuple[dict[str, float], dict]:
    """Ajuste GLOBAL (un único conjunto para todos los ℓ) contra energías NIST.

    Se conserva por compatibilidad. Para átomos pesados (K, Rb, Cs) sobre
    malla 100% lineal, prefiera :func:`refit_marinescu_params_per_l`, que
    permite que cada canal ℓ tenga su propia 5-upla.

    El objetivo del ajuste son ahora las energías NIST absolutas
    (E_nist_relativa − E_ion_H). Las energías
    QDT introducen un error sistemático adicional que no debe sumarse al error del modelo.
    """

    target_pairs: list[tuple[int, int, float]] = []
    for (n, l), E_rel in levels_relative.items():
        E_abs = E_rel - E_ionization_H
        if E_abs >= 0:
            continue
        target_pairs.append((n, l, E_abs))
    if not target_pairs:
        raise ValueError("No hay niveles NIST ligados válidos para ajustar.")
    target_pairs = target_pairs[:n_pairs_max]

    n_max = max(p[0] for p in target_pairs) + 1

    def loss_fn(params: np.ndarray) -> float:
        a1, a2, a3, a4, r_c = params
        if r_c <= 0 or a1 <= 0 or a2 <= 0:
            return 1e10
        total_loss = 0.0
        for n, l, E_target in target_pairs:
            r = build_radial_mesh(n_max=n, Z_c=Z_c, n_points=200000)
            p = PseudoPotentialParams(
                Z=Z, Z_c=Z_c, a1=a1, a2=a2, a3=a3, a4=a4,
                alpha_c=alpha_c_fixed, r_c=r_c, l=l,
            )
            try:
                sol = find_bound_state_near(
                    n, l, r, lambda rr: V_eff_marinescu_gen(rr, p),
                    E_guess=E_target, window_frac=0.30, n_scan=15,
                )
                if not sol.converged:
                    total_loss += 0.05
                else:
                    total_loss += (sol.E - E_target) ** 2
            except Exception:
                total_loss += 0.05
        return total_loss

    if initial_guess is None:
        initial_guess = (3.0, 1.5, 0.5, 0.1, max(1.0 / Z_c, 0.5))

    result = minimize(
        loss_fn, x0=np.array(initial_guess), method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": max_iter, "adaptive": True, "disp": verbose},
    )

    a1, a2, a3, a4, r_c = result.x
    params_dict = {
        "a1": float(a1), "a2": float(a2), "a3": float(a3), "a4": float(a4),
        "alpha_c": float(alpha_c_fixed), "r_c": float(r_c),
    }
    info = {"success": result.success, "loss": float(result.fun), "n_iter": result.nit}
    return params_dict, info


# ---------------------------------------------------------------------------
# Re-ajuste por canal ℓ
# ---------------------------------------------------------------------------


_PARAM_BOUNDS: tuple[tuple[float, float], ...] = (
    (0.5, 12.0),    
    (0.3, 5.0), 
    (-100.0, 100.0),
    (-2.0, 2.0),    
    (0.1, 10.0),    
)


def _clip_to_box(x: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(min(max(xi, lo), hi) for xi, (lo, hi) in zip(x, _PARAM_BOUNDS))


def _refit_one_l(
    Z: int, Z_c: int, alpha_c: float, l: int,
    target_states: list[tuple[int, float]],
    initial: tuple[float, float, float, float, float],
    n_points: int,
    max_iter: int,
    fixed_a1_a2_a4: tuple[float, float, float] | None = None,
) -> tuple[tuple[float, float, float, float, float], float, bool]:
    """Ajusta los 5 parámetros (a1, a2, a3, a4, r_c) para un único canal ℓ
    contra los niveles NIST absolutos `target_states = [(n, E_abs), ...]`.

    Si ``fixed_a1_a2_a4`` se provee como ``(a1_fix, a2_fix, a4_fix)``, el
    ajuste se reduce a sólo dos incógnitas (a3, r_c), útil para canales
    ℓ ≥ 2, donde típicamente hay sólo 2 puntos NIST y un ajuste a 5
    parámetros sobreajustaría. La idea física: a1, a2 describen la
    geometría del core de electrones internos (rígido, no depende de ℓ),
    por lo que pueden heredarse del canal p. Se fija a4 = 0 porque la
    barrera centrífuga aleja al electrón de valencia del core en
    estados d/f y la corrección polinomial fina deja de ser necesaria.

    Returns (params, loss, success).
    """
    reduced = fixed_a1_a2_a4 is not None
    if reduced:
        a1_fix, a2_fix, a4_fix = fixed_a1_a2_a4

    def loss(x: np.ndarray) -> float:
        if reduced:
            a3, r_c = float(x[0]), float(x[1])
            a1, a2, a4 = a1_fix, a2_fix, a4_fix
            full = np.array([a1, a2, a3, a4, r_c])
        else:
            a1, a2, a3, a4, r_c = (float(v) for v in x)
            full = np.asarray(x)

        pen = 0.0
        if reduced:
            # caja en a3 y r_c
            for xi, (lo, hi) in [(a3, _PARAM_BOUNDS[2]), (r_c, _PARAM_BOUNDS[4])]:
                if xi < lo:
                    pen += (lo - xi) ** 2
                elif xi > hi:
                    pen += (xi - hi) ** 2
        else:
            for xi, (lo, hi) in zip(full, _PARAM_BOUNDS):
                if xi < lo:
                    pen += (lo - xi) ** 2
                elif xi > hi:
                    pen += (xi - hi) ** 2

        if a1 <= 0 or a2 <= 0 or r_c <= 0:
            return 1e3 + pen

        total = 0.0
        for n, E_target in target_states:
            r = build_radial_mesh(n_max=n, Z_c=Z_c, n_points=n_points)
            p = PseudoPotentialParams(
                Z=Z, Z_c=Z_c, a1=a1, a2=a2, a3=a3, a4=a4,
                alpha_c=alpha_c, r_c=r_c, l=l,
            )
            try:
                sol = find_bound_state_near(
                    n, l, r, lambda rr: V_eff_marinescu_gen(rr, p),
                    E_guess=E_target, window_frac=0.30, n_scan=15,
                )
                if not sol.converged or not math.isfinite(sol.E):
                    total += 0.05
                else:
                    total += (sol.E - E_target) ** 2
            except Exception:
                total += 0.05
        return total + pen

    if reduced:
        # Semilla reducida: (a3, r_c) de la 5-tupla inicial.
        x0 = np.array([initial[2], initial[4]], dtype=float)
    else:
        x0 = np.array(_clip_to_box(tuple(initial)), dtype=float)
    contador_iter = [0] 

    def reporte_progreso(xk):
        contador_iter[0] += 1
        print(f"    ⏳ Nelder-Mead (canal ℓ={l}): iteración {contador_iter[0]} / {max_iter} evaluada...", end="\r", flush=True)

    res = minimize(
        loss, x0=x0, method="Nelder-Mead",
        options={"xatol": 5e-4, "fatol": 1e-10,
                 "maxiter": max_iter, "adaptive": True, "disp": False},
        callback=reporte_progreso
    )
    
    print(" " * 80, end="\r")


    if reduced:
        a3_fit, r_c_fit = float(res.x[0]), float(res.x[1])
        params = (a1_fix, a2_fix, a3_fit, a4_fix, r_c_fit)
    else:
        params = _clip_to_box(tuple(float(v) for v in res.x))
    return params, float(res.fun), bool(res.success)


def refit_marinescu_params_per_l(
    Z: int,
    Z_c: int,
    levels_relative: dict[tuple[int, int], float],
    E_ionization_H: float,
    alpha_c_fixed: float,
    initial_guess_per_l: dict[int, tuple[float, float, float, float, float]] | None = None,
    n_points: int = 200000,
    max_iter: int = 120,
    l_default: tuple[float, float, float, float, float] = (3.0, 1.5, -1.0, 0.0, 1.0),
    verbose: bool = True,
) -> tuple[dict, dict]:
    """Re-ajusta el pseudopotencial Marinescu-Dalgarno con **una 5-upla por
    canal ℓ**, contra los niveles NIST absolutos y sobre nuestra malla lineal.

    Mismo esquema que el paper original (Tabla I) pero con dos diferencias:

      1. El objetivo del ajuste son las energías NIST directamente — no las
         energías QDT extrapoladas (que añadirían un error sistemático).
      2. Los parámetros se acotan a `_PARAM_BOUNDS` para evitar pozos muy
         singulares cerca del origen que no se resuelven sobre malla lineal.

    Returns:
        (tabulated_params, info)
        ``tabulated_params`` tiene el mismo formato que ``MARINESCU_PARAMS[sym]``:
            {"alpha_c": float, 0: (a1,..,r_c), 1: (...), 2: (...), 3: (...)}
        Replica ℓ=3 para ℓ ≥ 4 (estados f, g, ... casi puros hidrogenoides).
    """
    # Agrupar niveles por canal ℓ
    by_l: dict[int, list[tuple[int, float]]] = {}
    for (n, l), E_rel in levels_relative.items():
        E_abs = E_rel - E_ionization_H
        if E_abs >= 0:
            continue
        by_l.setdefault(l, []).append((n, E_abs))

    if not by_l:
        raise ValueError("No hay niveles NIST ligados para ajustar (refit per ℓ).")

    if initial_guess_per_l is None:
        initial_guess_per_l = {}

    new_params: dict = {"alpha_c": float(alpha_c_fixed)}
    info_per_l: dict[int, dict] = {}
    total_loss = 0.0

    ls_with_data = sorted(by_l.keys())
    # Procesamos primero ℓ=0, luego ℓ=1, luego ℓ≥2: los canales d/f heredarán
    # a1, a2 del canal p (rigidez del core) y fijarán a4=0 (la barrera centrífuga
    # aleja al electrón de valencia del core, la corrección polinomial fina
    # deja de ser necesaria). Sólo quedan (a3, r_c) como libres en ℓ≥2.
    for l in ls_with_data:
        init_seed = (
            initial_guess_per_l.get(l)
            or initial_guess_per_l.get(min(l, 3))
            or l_default
        )

        fixed_a1_a2_a4: tuple[float, float, float] | None = None
        if l >= 2 and 1 in new_params:
            a1_p, a2_p, _a3_p, _a4_p, _rc_p = new_params[1]
            fixed_a1_a2_a4 = (a1_p, a2_p, 0.0)

        N_max = 5
        targets = by_l[l][:N_max]
        params, fval, success = _refit_one_l(
            Z=Z, Z_c=Z_c, alpha_c=alpha_c_fixed, l=l,
            target_states=targets, initial=init_seed,
            n_points=n_points, max_iter=max_iter,
            fixed_a1_a2_a4=fixed_a1_a2_a4,
        )
        new_params[l] = params
        info_per_l[l] = {
            "loss": fval, "success": success, "n_targets": len(targets),
            "reduced_mode": fixed_a1_a2_a4 is not None,
        }
        total_loss += fval

        if verbose:
            a1, a2, a3, a4, r_c = params
            mode_tag = "(a3,r_c only)" if fixed_a1_a2_a4 is not None else "(full 5-param)"
            print(f"    ℓ={l}: loss={fval:.3e}, params=(a1={a1:.3f}, a2={a2:.3f}, "
                  f"a3={a3:.3f}, a4={a4:.3f}, r_c={r_c:.3f})  "
                  f"[{len(targets)} estados] {mode_tag}")

    # Cubrir ℓ que no tienen datos NIST (típicamente ℓ ≥ 3 o ℓ = 2 a veces):
    # reusar el parámetro del ℓ más alto disponible. Para ℓ ≥ 4 también
    # se replica ℓ=3 si existe.
    max_l_with_data = max(ls_with_data)
    for l in range(0, 6):
        if l not in new_params:
            ref_l = max_l_with_data if l > max_l_with_data else min(
                (lk for lk in ls_with_data if lk >= l), default=max_l_with_data
            )
            new_params[l] = new_params[ref_l]
            info_per_l[l] = {"loss": float("nan"), "success": True, "replicated_from": ref_l}

    info = {
        "method": "per_l_nist_direct",
        "per_l": info_per_l,
        "total_loss": float(total_loss),
        "n_points": int(n_points),
    }
    return new_params, info


# =============================================================================
# CLASE DE ALTO NIVEL: ÁTOMO ALCALINO
# =============================================================================

@dataclass
class AlkaliAtom:
    symbol: str
    Z: int
    Z_c: int = 1                  
    n0: int = 1                   
    l0: int = 0
    params: dict = field(default_factory=dict)
    is_tabulated: bool = False
    tabulated_params: dict = field(default_factory=dict)
    nist_levels: dict = field(default_factory=dict)
    E_ionization_H: float = 0.0
    alpha_c: float = 0.0
    fit_info: dict = field(default_factory=dict)

    def V_eff(self, r: np.ndarray, l: int) -> np.ndarray:
        if self.is_tabulated:
            l_key = min(l, 3)
            a1, a2, a3, a4, r_c = self.tabulated_params[l_key]
            p = PseudoPotentialParams(
                Z=self.Z, Z_c=self.Z_c, a1=a1, a2=a2, a3=a3, a4=a4,
                alpha_c=self.alpha_c, r_c=r_c, l=l,
            )
        else:
            p = PseudoPotentialParams(
                Z=self.Z, Z_c=self.Z_c,
                a1=self.params["a1"], a2=self.params["a2"], a3=self.params["a3"], a4=self.params["a4"],
                alpha_c=self.alpha_c, r_c=self.params["r_c"], l=l,
            )
        return V_eff_marinescu_gen(r, p)

    def dV_eff(self, r: np.ndarray, l: int) -> np.ndarray:
        if self.is_tabulated:
            l_key = min(l, 3)
            a1, a2, a3, a4, r_c = self.tabulated_params[l_key]
            p = PseudoPotentialParams(
                Z=self.Z, Z_c=self.Z_c, a1=a1, a2=a2, a3=a3, a4=a4,
                alpha_c=self.alpha_c, r_c=r_c, l=l,
            )
        else:
            p = PseudoPotentialParams(
                Z=self.Z, Z_c=self.Z_c,
                a1=self.params["a1"], a2=self.params["a2"], a3=self.params["a3"], a4=self.params["a4"],
                alpha_c=self.alpha_c, r_c=self.params["r_c"], l=l,
            )
        return dV_eff_marinescu_gen(r, p)

    def solve_state(self, n: int, l: int, n_points: int = 200000,
                    use_nist_seed: bool = True) -> RadialSolution:
        """Resuelve el estado |n, l>.

        Si hay datos NIST para (n, l) y use_nist_seed=True (default), usa el
        valor experimental como semilla y refina en una ventana estrecha.
        Esto reduce las llamadas al Wronskiano de ~250 a ~30 por estado.

        Si no hay semilla NIST, hace un barrido grueso completo y selecciona
        la raíz con el número correcto de nodos n - l - 1.

        La malla se dimensiona específicamente para el estado solicitado
        (`build_radial_mesh(n_max=n, ...)`), con `n_points=200000` por defecto,
        que es la misma malla usada por `refit_marinescu_params_per_l`: así
        los parámetros re-ajustados se aplican sobre exactamente la misma
        discretización contra la que fueron optimizados.

        Nota sobre nodos en pseudopotenciales: para átomos pesados (Cs, Rb)
        el pseudopotencial Marinescu absorbe los estados de core en el potencial,
        por lo que el conteo de nodos del eigenestado físico puede no ser
        n - l - 1. Con semilla NIST aceptamos la energía cercana al valor
        experimental sin imponer el conteo hidrogenoide.
        """
        r = build_radial_mesh(n_max=n, Z_c=self.Z_c, n_points=n_points)
        V_func = lambda rr: self.V_eff(rr, l)

        sol = None
        if use_nist_seed and (n, l) in self.nist_levels:
            E_rel = self.nist_levels[(n, l)]
            E_guess = E_rel - self.E_ionization_H  
            sol_try = find_bound_state_near(
                n, l, r, V_func, E_guess=E_guess,
                window_frac=0.4, n_scan=25,
            )
            if sol_try.converged and abs(sol_try.E - E_guess) < 0.5 * abs(E_guess):
                sol = sol_try

        if sol is None:
            sol = find_bound_state(n, l, r, V_func, Z_c=self.Z_c)

        if sol.converged:
            sol.xi_nl = compute_xi_nl(sol, lambda rr: self.dV_eff(rr, l))
        return sol

    def solve_subspace(self, n_values: list[int] | None = None) -> list[RadialSolution]:
        if n_values is None:
            n_values = [self.n0, self.n0 + 1]
        results: list[RadialSolution] = []
        for n in n_values:
            for l in range(0, n):
                if n == self.n0 and l < self.l0:
                    continue
                results.append(self.solve_state(n, l))
        return results

    @classmethod
    def from_tabulated(cls, symbol: str) -> "AlkaliAtom":
        if symbol not in MARINESCU_PARAMS or symbol not in ALKALI_GROUND_STATES:
            raise ValueError(f"Datos insuficientes para {symbol}.")
        gs = ALKALI_GROUND_STATES[symbol]
        return cls(
            symbol=symbol, Z=gs["Z"], Z_c=1, n0=gs["n0"], l0=gs["l0"],
            is_tabulated=True,
            tabulated_params=MARINESCU_PARAMS[symbol],
            nist_levels=NIST_REFERENCE_LEVELS.get(symbol, {}),
            E_ionization_H=gs["E_ion_H"],
            alpha_c=MARINESCU_PARAMS[symbol]["alpha_c"],
        )

    @classmethod
    def from_nist_fit(
        cls, symbol: str, levels_relative: dict[tuple[int, int], float] | None = None,
        alpha_c_fixed: float | None = None, verbose: bool = True,
        per_l: bool = True,
        n_points: int = 200000,
        max_iter: int = 100,
    ) -> "AlkaliAtom":
        """Construye un AlkaliAtom ajustando el pseudopotencial Marinescu-Dalgarno
        contra los niveles NIST de la especie.

        Por defecto (``per_l=True``) usa una 5-upla independiente por canal ℓ,
        igual que la Tabla I del paper original. Esto es esencial para reproducir
        las energías de los alcalinos pesados (K, Rb, Cs) sobre nuestra malla
        100% lineal, donde los valores tabulados originales no aplican.

        Si se proveen parámetros tabulados de Marinescu en `MARINESCU_PARAMS`
        para el símbolo, se usan como semilla inicial para Nelder-Mead.
        """
        if symbol not in ALKALI_GROUND_STATES:
            raise ValueError(f"No hay configuración base para {symbol}.")
        gs = ALKALI_GROUND_STATES[symbol]

        if levels_relative is None:
            levels_relative = NIST_REFERENCE_LEVELS.get(symbol, {})
        if alpha_c_fixed is None:
            alpha_c_fixed = MARINESCU_PARAMS.get(symbol, {}).get("alpha_c", 0.1)

        if per_l:
            # Semilla por ℓ: usar los parámetros tabulados de Marinescu si existen.
            initial_per_l = None
            if symbol in MARINESCU_PARAMS:
                initial_per_l = {
                    l: MARINESCU_PARAMS[symbol][l]
                    for l in MARINESCU_PARAMS[symbol] if isinstance(l, int)
                }
            if verbose:
                print(f"  [refit per ℓ] {symbol}: ajustando {len(levels_relative)} "
                      f"niveles NIST, n_points={n_points}")
            tabulated, info = refit_marinescu_params_per_l(
                Z=gs["Z"], Z_c=1, levels_relative=levels_relative,
                E_ionization_H=gs["E_ion_H"], alpha_c_fixed=alpha_c_fixed,
                initial_guess_per_l=initial_per_l, n_points=n_points,
                max_iter=max_iter, verbose=verbose,
            )
            return cls(
                symbol=symbol, Z=gs["Z"], Z_c=1, n0=gs["n0"], l0=gs["l0"],
                is_tabulated=True, tabulated_params=tabulated,
                nist_levels=levels_relative,
                E_ionization_H=gs["E_ion_H"], alpha_c=alpha_c_fixed, fit_info=info,
            )

        params, info = fit_pseudopotential_params(
            Z=gs["Z"], Z_c=1, levels_relative=levels_relative,
            E_ionization_H=gs["E_ion_H"], alpha_c_fixed=alpha_c_fixed, verbose=verbose,
            max_iter=max_iter,
        )
        return cls(
            symbol=symbol, Z=gs["Z"], Z_c=1, n0=gs["n0"], l0=gs["l0"],
            params=params, is_tabulated=False, nist_levels=levels_relative,
            E_ionization_H=gs["E_ion_H"], alpha_c=alpha_c_fixed, fit_info=info,
        )

    @classmethod
    def from_species(cls, symbol: str,
                     nist_levels: dict[tuple[int, int], float] | None = None,
                     verbose: bool = False,
                     prefer_tabulated: bool = False,
                     n_points: int = 200000,
                     max_iter: int = 100) -> "AlkaliAtom":
        """Factoría universal que cubre cualquier especie del catálogo periódico.

        Estrategia automática (en orden):

            1) Si ``prefer_tabulated=True`` y hay parámetros Marinescu tabulados
               para la especie neutra, usa ``from_tabulated`` (parámetros del
               paper original, válidos sobre malla logarítmica).
            2) En caso contrario, si hay datos NIST suficientes, hace un
               **re-ajuste por canal ℓ** del pseudopotencial Marinescu sobre
               nuestra malla lineal. **Esta es la ruta por defecto** y
               la recomendada en este proyecto: la única forma de obtener
               errores < 1500 cm⁻¹ para K, Rb, Cs.
            3) Si tampoco hay datos NIST, advierte y entrega un átomo con
               parámetros suaves (resultados aproximados).

        El símbolo puede ser cualquier entrada del catálogo
        (`periodic_catalog.py`) e incluir cationes: 'Li', 'Ca+', 'B2+', 'Cu', etc.
        """
        from periodic_catalog import get_species
        sp = get_species(symbol)

        # Caso 1: ruta clásica de parámetros tabulados Marinescu
        if prefer_tabulated and sp.symbol in MARINESCU_PARAMS and sp.q == 0:
            atom = cls.from_tabulated(sp.symbol)
            if nist_levels is not None:
                atom.nist_levels = nist_levels
            return atom

        # Caso 2: ajuste por canal ℓ contra NIST
        levels = nist_levels or NIST_REFERENCE_LEVELS.get(sp.symbol, {})
        if levels:
            initial_per_l = None
            if sp.symbol in MARINESCU_PARAMS and sp.q == 0:
                initial_per_l = {
                    l: MARINESCU_PARAMS[sp.symbol][l]
                    for l in MARINESCU_PARAMS[sp.symbol] if isinstance(l, int)
                }
            if verbose:
                print(f"  [refit per ℓ] {sp.symbol} (Z={sp.Z}, q={sp.q}): "
                      f"ajustando {len(levels)} niveles NIST")
            tabulated, info = refit_marinescu_params_per_l(
                Z=sp.Z, Z_c=sp.Z_c, levels_relative=levels,
                E_ionization_H=sp.E_ion_H, alpha_c_fixed=sp.alpha_c,
                initial_guess_per_l=initial_per_l, n_points=n_points,
                max_iter=max_iter, verbose=verbose,
            )
            return cls(
                symbol=sp.symbol, Z=sp.Z, Z_c=sp.Z_c, n0=sp.n0, l0=sp.l0,
                is_tabulated=True, tabulated_params=tabulated,
                nist_levels=levels,
                E_ionization_H=sp.E_ion_H, alpha_c=sp.alpha_c, fit_info=info,
            )

        # Caso 3: sin datos NIST → parámetros suaves de fallback (con aviso).
        warnings.warn(
            f"Sin datos NIST para {symbol}: usando parámetros suaves "
            f"hidrogenoides (precisión limitada). Considere proveer "
            f"nist_levels={{(n,l): E_rel}}.",
            stacklevel=2,
        )
        return cls(
            symbol=sp.symbol, Z=sp.Z, Z_c=sp.Z_c, n0=sp.n0, l0=sp.l0,
            params={"a1": 3.0, "a2": 1.5, "a3": 0.0, "a4": 0.0, "r_c": max(1.0/sp.Z_c, 0.5)},
            is_tabulated=False, nist_levels={},
            E_ionization_H=sp.E_ion_H, alpha_c=sp.alpha_c,
        )

    @classmethod
    def make_hydrogenic(cls, Z: int) -> "AlkaliAtom":
        """Construye un átomo hidrogenoide puro (1 electrón) de carga nuclear Z.
        Aplicable a H, He+, Li²+, Be³+, ..., U⁹¹+, etc.
        El potencial es exactamente -Z/r (sin polarización ni apantallamiento).
        """
        from periodic_catalog import make_hydrogenic_species
        sp = make_hydrogenic_species(Z)
        return cls(
            symbol=sp.symbol, Z=Z, Z_c=Z, n0=1, l0=0,
            params={"a1": 5.0, "a2": 1.5, "a3": 0.0, "a4": 0.0, "r_c": 1.0},
            is_tabulated=False, nist_levels={},
            E_ionization_H=Z**2 / 2.0, alpha_c=0.0,
        )


# =============================================================================
# UTILIDADES DE VALIDACIÓN
# =============================================================================

def validate_against_nist(atom: AlkaliAtom, n_values: list[int] | None = None) -> "list[dict]":
    if n_values is None:
        n_values = sorted(set(nl[0] for nl in atom.nist_levels.keys()))
    out: list[dict] = []
    for (n, l), E_rel in atom.nist_levels.items():
        if n_values is not None and n not in n_values:
            continue
        E_nist_abs = E_rel - atom.E_ionization_H  
        sol = atom.solve_state(n, l)
        E_calc = sol.E
        out.append({
            "symbol": atom.symbol, "Z": atom.Z, "n": n, "l": l,
            "E_calc_H": E_calc, "E_nist_H": E_nist_abs,
            "diff_H": E_calc - E_nist_abs if math.isfinite(E_calc) else float("nan"),
            "diff_cm-1": (E_calc - E_nist_abs) * HARTREE_TO_CM if math.isfinite(E_calc) else float("nan"),
            "xi_nl_H": sol.xi_nl, "converged": sol.converged, "n_nodes": sol.n_nodes,
        })
    return out
