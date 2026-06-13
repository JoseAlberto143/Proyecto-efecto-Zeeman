"""
=============================================================================
CATÁLOGO DE ÁTOMOS TIPO ALCALINO DE LA TABLA PERIÓDICA
=============================================================================

Un "átomo tipo alcalino" es cualquier especie atómica con UN SOLO electrón de
valencia fuera de una capa cerrada tipo gas noble (o pseudo-cerrada). Este
catálogo lista todas las especies (Z, q) razonablemente accesibles, indicando:

    - símbolo y carga
    - número atómico Z
    - n y l del electrón de valencia
    - energía de ionización experimental (Hartree)
    - polarizabilidad dipolar del core α_c (Hartree·Bohr^3) cuando se conoce
    - nombre del gas noble que forma el core

Categorías incluidas:
    A. Group 1 neutros: H, Li, Na, K, Rb, Cs, Fr
    B. Group 2 +1 (alcalinotérreos ionizados): Be+, Mg+, Ca+, Sr+, Ba+, Ra+
    C. Group 13 +2: B²+, Al²+, Ga²+, In²+, Tl²+
    D. Coinage metals (Cu, Ag, Au): se tratan como tipo alcalino con core d¹⁰
    E. Iones hidrogenoides (He+, Li²+, ...): tratados con factory aparte.

Fuentes de los datos:
- Energías de ionización: NIST ASD (cm⁻¹ → Hartree con 1 H = 219474.63 cm⁻¹)
- Polarizabilidades de core α_c: paper Marinescu et al. PRA 49, 982 (1994)
  para He/Ne/Ar/Kr/Xe cores; otras de revisiones de Mitroy et al. (2010).
=============================================================================
"""
from __future__ import annotations
from dataclasses import dataclass


.
CORE_POLARIZABILITIES = {
    "none":   0.0,       # H (sin core)
    "He":     0.1923,    # Marinescu (Li, Be+, B²+, ...)
    "Ne":     0.9448,    # Marinescu (Na, Mg+, Al²+, ...)
    "Ar":     5.3310,    # Marinescu (K, Ca+, ...)
    "Kr":     9.0760,    # Marinescu (Rb, Sr+, Y²+, ...)
    "Xe":    15.6440,    # Marinescu (Cs, Ba+, La²+, ...)
    "Rn":    33.18,      # estimado (Fr, Ra+, Ac²+, ...)
    "Cu_d10": 5.36,      # core 3d¹⁰ de Cu
    "Ag_d10": 7.74,      # core 4d¹⁰ de Ag
    "Au_d10": 12.0,      # core 5d¹⁰ de Au
}


@dataclass
class AlkaliSpecies:
    """Descripción de una especie atómica tipo alcalino."""
    symbol: str          
    Z: int               
    q: int               
    n0: int              
    l0: int              
    E_ion_H: float       
    core: str            
    alpha_c: float       
    notes: str = ""

    @property
    def Z_c(self) -> int:
        """Carga residual sentida asintóticamente por la valencia."""
        return self.q + 1


# =============================================================================
# CATÁLOGO COMPLETO
# =============================================================================

ALKALI_LIKE_SPECIES: dict[str, AlkaliSpecies] = {}


def _add(symbol, Z, q, n0, l0, E_ion_cm, core, notes=""):
    """Helper: añade una especie al catálogo, convirtiendo cm⁻¹ a Hartree."""
    HARTREE_TO_CM = 219474.63136320
    E_ion_H = E_ion_cm / HARTREE_TO_CM
    ALKALI_LIKE_SPECIES[symbol] = AlkaliSpecies(
        symbol=symbol, Z=Z, q=q, n0=n0, l0=l0, E_ion_H=E_ion_H,
        core=core, alpha_c=CORE_POLARIZABILITIES.get(core, 0.0), notes=notes,
    )


# Metales alcalinos clásicos
_add("H",   1,  0, 1, 0, 109678.77, "none", "potencial coulombiano puro")
_add("Li",  3,  0, 2, 0,  43487.15, "He",   "parámetros Marinescu tabulados")
_add("Na", 11,  0, 3, 0,  41449.45, "Ne",   "parámetros Marinescu tabulados")
_add("K",  19,  0, 4, 0,  35009.81, "Ar",   "parámetros Marinescu tabulados")
_add("Rb", 37,  0, 5, 0,  33690.81, "Kr",   "parámetros Marinescu tabulados")
_add("Cs", 55,  0, 6, 0,  31406.47, "Xe",   "parámetros Marinescu tabulados")
_add("Fr", 87,  0, 7, 0,  32848.87, "Rn",   "experimental por relojes atómicos")

# Alcalinotérreos ionizados

_add("Be+",  4, 1, 2, 0, 146882.86, "He", "ión hidrogenoide ligero")
_add("Mg+", 12, 1, 3, 0, 121267.61, "Ne", "ión común en espectroscopía")
_add("Ca+", 20, 1, 4, 0,  95751.87, "Ar", "candidato a reloj óptico")
_add("Sr+", 38, 1, 5, 0,  88964.00, "Kr", "reloj óptico, 88Sr+")
_add("Ba+", 56, 1, 6, 0,  80686.30, "Xe", "reloj óptico, computación cuántica")
_add("Ra+", 88, 1, 7, 0,  81842.20, "Rn", "EDM / física fundamental")

# Grupo 13 +2

_add("B2+",   5, 2, 2, 0, 305931.00, "He", "ión Be-like de boro")
_add("Al2+", 13, 2, 3, 0, 229453.95, "Ne", "ión Mg-like de aluminio")
_add("Ga2+", 31, 2, 4, 0, 247389.00, "Ar", "ión Ca-like de galio")
_add("In2+", 49, 2, 5, 0, 226020.00, "Kr", "ión Sr-like de indio")
_add("Tl2+", 81, 2, 6, 0, 240634.00, "Xe", "ión Ba-like de talio")

# Metales de acuñación 
_add("Cu", 29, 0, 4, 0,  62317.46, "Cu_d10", "[Ar] 3d¹⁰ 4s¹, núcleo no es gas noble")
_add("Ag", 47, 0, 5, 0,  61106.45, "Ag_d10", "[Kr] 4d¹⁰ 5s¹")
_add("Au", 79, 0, 6, 0,  74409.11, "Au_d10", "[Xe] 4f¹⁴ 5d¹⁰ 6s¹, fuertes correcciones relativistas")


# =============================================================================
# UTILIDADES
# =============================================================================

def get_species(symbol: str) -> AlkaliSpecies:
    """Devuelve la descripción de la especie. Acepta 'Li', 'Ca+', 'B2+', etc."""
    if symbol not in ALKALI_LIKE_SPECIES:
        available = ", ".join(sorted(ALKALI_LIKE_SPECIES.keys()))
        raise ValueError(
            f"Especie '{symbol}' no está en el catálogo.\n"
            f"Disponibles: {available}\n"
            f"Para iones hidrogenoides (1 electrón total) use make_hydrogenic(Z)."
        )
    return ALKALI_LIKE_SPECIES[symbol]


def list_species(category: str = "all") -> list[str]:
    """Lista los símbolos del catálogo. category ∈ {'all', 'neutral', 'cation'}."""
    if category == "neutral":
        return [s for s, sp in ALKALI_LIKE_SPECIES.items() if sp.q == 0]
    if category == "cation":
        return [s for s, sp in ALKALI_LIKE_SPECIES.items() if sp.q > 0]
    return list(ALKALI_LIKE_SPECIES.keys())


def make_hydrogenic_species(Z: int) -> AlkaliSpecies:
    """Construye sobre la marcha la especie hidrogenoide H-like de carga Z-1.
    Útil para He+, Li²+, Be³+, ..., U⁹¹+ — todos con UN electrón.
    """
    if Z < 1:
        raise ValueError(f"Z={Z} inválido para especie hidrogenoide.")
    E_ion = (Z ** 2) / 2.0
    symbol = f"Z={Z}_H-like"
    if Z == 1:
        symbol = "H"
    else:
        symbol = f"Z={Z}^{Z-1}+"
    return AlkaliSpecies(
        symbol=symbol, Z=Z, q=Z - 1, n0=1, l0=0,
        E_ion_H=E_ion, core="none", alpha_c=0.0,
        notes="Especie hidrogenoide pura: 1 electrón, potencial -Z/r",
    )


if __name__ == "__main__":
    
    print(f"Total de especies tipo alcalino tabuladas: {len(ALKALI_LIKE_SPECIES)}")
    print(f"\n{'sym':<7}{'Z':<4}{'q':<4}{'n0':<4}{'l0':<4}"
          f"{'E_ion (Ht)':<13}{'core':<10}{'α_c':<8}{'notas'}")
    print("-" * 90)
    for sym, sp in ALKALI_LIKE_SPECIES.items():
        print(f"{sym:<7}{sp.Z:<4}{sp.q:<4}{sp.n0:<4}{sp.l0:<4}"
              f"{sp.E_ion_H:<13.5f}{sp.core:<10}{sp.alpha_c:<8.3f}{sp.notes}")
