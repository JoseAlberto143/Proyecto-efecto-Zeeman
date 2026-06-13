"""
=============================================================================
NÚCLEO NUMÉRICO OPTIMIZADO CON NUMBA JIT
=============================================================================

Implementa los bucles Numerov outward/inward y la búsqueda del Wronskiano en
funciones independientes que Numba puede compilar a código máquina. Esto da
una mejora de 10x - 30x sobre los mismos bucles en Python, manteniendo
exactamente la misma lógica numérica del archivo original.

Uso:
    from numerov_fast import shoot_wronskian_fast
    W, P_combined, m_match = shoot_wronskian_fast(E, r, V, l)

=============================================================================
"""
from __future__ import annotations
import math
import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def wrapper(func):
            return func
        return wrapper


# =============================================================================
# NÚCLEO NUMEROV 
# =============================================================================

@njit(fastmath=True)
def _numerov_outward_core(r: np.ndarray, V: np.ndarray, E: float,
                          l: int, m_match: int) -> np.ndarray:
    """Integración Numerov de afuera-hacia-adentro desde r=0.
    Implementación exacta del avance, vectorizable por Numba."""
    N = r.shape[0]
    P = np.zeros(N)
    h = r[1] - r[0]
    h2_12 = h * h / 12.0
    f = np.empty(N)
    for i in range(N):
        ri = r[i] if r[i] > 1e-12 else 1e-12
        f[i] = l * (l + 1) / (ri * ri) + 2.0 * V[i] - 2.0 * E

    # Condiciones iniciales
    P[0] = 0.0
    P[1] = h ** (l + 1) if l >= 0 else h

    for i in range(1, m_match):
        c_curr = 2.0 * (1.0 + 5.0 * h2_12 * f[i]) * P[i]
        c_prev = (1.0 - h2_12 * f[i - 1]) * P[i - 1]
        denom = 1.0 - h2_12 * f[i + 1]
        if abs(denom) < 1e-14:
            denom = 1e-14
        P[i + 1] = (c_curr - c_prev) / denom

        if abs(P[i + 1]) > 1e20:
            scale = 1e20 / abs(P[i + 1])
            for k in range(i + 2):
                P[k] *= scale

    return P


@njit(fastmath=True)
def _numerov_inward_core(r: np.ndarray, V: np.ndarray, E: float,
                         l: int, m_match: int) -> np.ndarray:
    """Integración Numerov de adentro-hacia-afuera desde r=r_max."""
    N = r.shape[0]
    P = np.zeros(N)
    h = r[1] - r[0]
    h2_12 = h * h / 12.0

    f = np.empty(N)
    for i in range(N):
        ri = r[i] if r[i] > 1e-12 else 1e-12
        f[i] = l * (l + 1) / (ri * ri) + 2.0 * V[i] - 2.0 * E
    if E < 0:
        kappa = math.sqrt(-2.0 * E)
    else:
        kappa = 1.0

    P[N - 1] = math.exp(-kappa * r[N - 1])
    P[N - 2] = math.exp(-kappa * r[N - 2])

    if P[N - 1] < 1e-50:
        P[N - 1] = 1e-50
        P[N - 2] = 1e-50 * math.exp(kappa * h)

    for i in range(N - 2, m_match, -1):
        c_curr = 2.0 * (1.0 + 5.0 * h2_12 * f[i]) * P[i]
        c_next = (1.0 - h2_12 * f[i + 1]) * P[i + 1]
        denom = 1.0 - h2_12 * f[i - 1]
        if abs(denom) < 1e-14:
            denom = 1e-14
        P[i - 1] = (c_curr - c_next) / denom

        if abs(P[i - 1]) > 1e20:
            scale = 1e20 / abs(P[i - 1])
            for k in range(i - 1, N):
                P[k] *= scale

    return P


@njit(fastmath=True)
def _find_turning_point_core(r: np.ndarray, V: np.ndarray, E: float, l: int) -> int:
    """Localiza el punto de retorno clásico: el último r donde E > V_eff_total(r)."""
    N = r.shape[0]
    last_pos = -1
    for i in range(N):
        ri = r[i] if r[i] > 1e-12 else 1e-12
        V_total = V[i] + l * (l + 1) / (2.0 * ri * ri)
        if E - V_total > 0:
            last_pos = i
    if last_pos < 0:
        return N // 2
    return last_pos


@njit(fastmath=True)
def _shoot_wronskian_core(E: float, r: np.ndarray, V: np.ndarray, l: int):
    """Núcleo JIT del cálculo del Wronskiano + función combinada."""
    N = r.shape[0]
    m_match = _find_turning_point_core(r, V, E, l)
    if m_match < 5:
        m_match = 5
    if m_match > N - 5:
        m_match = N - 5

    P_out = _numerov_outward_core(r, V, E, l, m_match + 2)
    P_in = _numerov_inward_core(r, V, E, l, m_match - 2)

    if abs(P_in[m_match]) < 1e-300:
        W = 1e300
        P_combined = P_out.copy()
        return W, P_combined, m_match

    scale = P_out[m_match] / P_in[m_match]
    # Formulación del Wronskiano
    W = (P_in[m_match + 1] * scale - P_out[m_match + 1]
         + P_out[m_match - 1] - P_in[m_match - 1] * scale)

    # Función radial combinada
    P_combined = P_out.copy()
    for i in range(m_match, N):
        P_combined[i] = P_in[i] * scale

    return W, P_combined, m_match


# =============================================================================
# API PÚBLICA
# =============================================================================

def shoot_wronskian_fast(E: float, r: np.ndarray, V: np.ndarray,
                         l: int) -> tuple[float, np.ndarray, int]:
    return _shoot_wronskian_core(float(E), np.ascontiguousarray(r, dtype=np.float64),
                                  np.ascontiguousarray(V, dtype=np.float64), int(l))


def warmup_jit():
    if not NUMBA_AVAILABLE:
        return
    r_dummy = np.linspace(0.01, 1.0, 50)
    V_dummy = -1.0 / r_dummy
    _ = shoot_wronskian_fast(-0.5, r_dummy, V_dummy, 0)
