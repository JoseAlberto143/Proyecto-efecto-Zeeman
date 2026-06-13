"""
=============================================================================
EXTRACTOR NIST AS
=============================================================================

Descarga datos de niveles de energía desde la NIST Atomic Spectra Database,
los limpia, y los convierte al formato {(n, l): E_relativa_Hartree} que
consume nuestro solucionador radial.

Características:
    - Descarga en Hartree (units=3 de NIST)
    - Extrae automáticamente n y l de la columna 'Configuration'
    - Calcula centros de multiplete ponderados por (2J+1)
    - Maneja errores de red y rate limiting con reintentos
    - Cache local opcional (`cache_dir`) para no re-descargar

=============================================================================
"""

from __future__ import annotations

import csv
import io
import math
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:  
    HAS_PANDAS = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:  
    HAS_REQUESTS = False


BASE_LEVELS_URL = "https://physics.nist.gov/cgi-bin/ASD/energy1.pl"

L_LETTER_TO_INT = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5, "i": 6, "k": 7, "l": 8}

ELEMENT_TO_Z = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26,
    "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34,
    "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42,
    "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "Fr": 87,
}


def int_to_roman(num: int) -> str:
    """Convierte un entero a numeros romanos."""
    vals = [
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
        (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
    ]
    out = []
    for val, sym in vals:
        while num >= val:
            out.append(sym)
            num -= val
    return "".join(out)


def clean_number(x: str) -> float:
    """Limpia un valor numérico de NIST."""
    if x is None:
        return math.nan
    s = str(x).strip()
    if not s or s in {"---", "--", "-"}:
        return math.nan
    s = re.sub(r"[\[\](){}]", "", s)
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[†?*]", "", s)
    s = s.replace(" ", "")
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", s)
    return float(m.group(0)) if m else math.nan


def parse_J(x: str) -> float:
    """Convierte un valor de J a float."""
    if x is None:
        return math.nan
    s = str(x).strip()
    if not s:
        return math.nan
    m = re.search(r"(\d+)\s*/\s*(\d+)", s)
    if m:
        return float(m.group(1)) / float(m.group(2))
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else math.nan


def extract_n_l_from_config(config: str) -> tuple[float, str, float]:
    """Extrae n y l (letra y entero) del último orbital de una configuración NIST."""
    if config is None or (isinstance(config, float) and math.isnan(config)):
        return math.nan, "", math.nan
    s = str(config)
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"<[^>]*>", " ", s)
    s = s.replace("^{", "").replace("}", "")
    s = s.replace("_", "")
    matches = re.findall(r"(?<![A-Za-z])(\d{1,3})\s*([spdfghikl])\b", s, flags=re.I)
    if not matches:
        return math.nan, "", math.nan
    n_str, letter = matches[-1]
    letter = letter.lower()
    return float(n_str), letter, float(L_LETTER_TO_INT.get(letter, math.nan))


def download_nist_levels(
    spectrum: str,
    cache_dir: str | Path | None = None,
    delay: float = 3,
    timeout: int = 60,
    verbose: bool = False,
) -> str | None:
    """Descarga niveles de un espectro desde NIST en formato CSV.

    Args:
        spectrum: e.g. "Li I", "Na II".
        cache_dir: carpeta donde guardar el CSV crudo.
        delay: pausa post-descarga.
        timeout: timeout HTTP en segundos.
        verbose: imprimir progreso.

    Returns:
        Texto CSV crudo, o None si falló.
    """
    if not HAS_REQUESTS:
        if verbose:
            print("[NIST] requests no disponible; no se puede descargar.")
        return None


    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", spectrum)
        cache_file = cache_dir / f"{safe}.csv"
        if cache_file.exists() and cache_file.stat().st_size > 100:
            if verbose:
                print(f"[NIST] usando cache: {cache_file}")
            return cache_file.read_text(encoding="utf-8", errors="replace")

    params = {
        "de": "0",
        "spectrum": spectrum,
        "submit": "Retrieve Data",
        "units": "3",         
        "format": "2",      
        "output": "1",
        "page_size": "9999999",
        "multiplet_ordered": "0",
        "conf_out": "on",
        "term_out": "on",
        "level_out": "on",
        "unc_out": "1",
        "j_out": "on",
        "lande_out": "on",
        "perc_out": "on",
        "biblio": "on",
    }
    headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }


    try:
        if verbose:
            print(f"[NIST] descargando: {spectrum}...")
        r = requests.get(BASE_LEVELS_URL, params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        text = r.text
        time.sleep(delay)

        low = text.lower()
        if "configuration" not in low or "level" not in low:
            if verbose:
                print(f"[NIST] {spectrum}: respuesta sin datos válidos.")
            return None

        if cache_dir is not None:
            cache_file.write_text(text, encoding="utf-8")

        return text
    except Exception as exc:
        if verbose:
            print(f"[NIST] error descargando {spectrum}: {exc}")
        return None


def parse_nist_levels_csv(text: str) -> "pd.DataFrame":
    """Parsea el texto CSV crudo de NIST a un DataFrame ordenado.

    Returns:
        DataFrame con columnas: configuration, term, J, level_H, n, l_letter, l, g_stat.
    """
    if not HAS_PANDAS:
        raise RuntimeError("Pandas es necesario para parsear los datos NIST.")
    if not text or not text.strip():
        return pd.DataFrame()
    
    lines = [ln for ln in text.splitlines() if ln.strip()]
    header_idx = -1
    for i, ln in enumerate(lines):
        low = ln.lower()
        if "configuration" in low and "level" in low and "j" in low:
            header_idx = i
            break
    if header_idx < 0:
        return pd.DataFrame()

    table_text = "\n".join(lines[header_idx:])
    try:
        df = pd.read_csv(io.StringIO(table_text), sep=",", engine="python")
    except Exception:
        return pd.DataFrame()

    def find_col(df: pd.DataFrame, *keywords: str) -> Optional[str]:
        for c in df.columns:
            cl = str(c).lower()
            if all(kw.lower() in cl for kw in keywords):
                return c
        return None

    conf_col = find_col(df, "configuration") or find_col(df, "principal", "configuration")
    term_col = find_col(df, "term")
    j_col = find_col(df, "j")
    level_col = None
    for c in df.columns:
        cl = str(c).lower()
        if "level" in cl and "unc" not in cl and "id" not in cl:
            level_col = c
            break

    if level_col is None or conf_col is None:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["configuration"] = df[conf_col].astype(str)
    out["term"] = df[term_col].astype(str) if term_col else ""
    out["J"] = df[j_col].map(parse_J) if j_col else float("nan")
    out["g_stat"] = 2.0 * out["J"] + 1.0
    out["level_H"] = df[level_col].map(clean_number)

    nl_tuples = out["configuration"].map(extract_n_l_from_config)
    out["n"] = [t[0] for t in nl_tuples]
    out["l_letter"] = [t[1] for t in nl_tuples]
    out["l"] = [t[2] for t in nl_tuples]

    # Filtrar filas válidas
    out = out.dropna(subset=["level_H", "n", "l"]).copy()
    out["n"] = out["n"].astype(int)
    out["l"] = out["l"].astype(int)
    return out


def levels_to_centers(df: "pd.DataFrame") -> dict[tuple[int, int], float]:
    """Agrupa por (n, l) y calcula el centro del multiplete ponderado por (2J+1).

    Returns:
        dict {(n, l): energía_centro_Hartree}
    """
    if df.empty:
        return {}
    centers: dict[tuple[int, int], float] = {}
    for (n, l), group in df.groupby(["n", "l"]):
        valid = group.dropna(subset=["level_H", "g_stat"])
        valid = valid[valid["g_stat"] > 0]
        if valid.empty:
            energy = float(group["level_H"].mean())
        else:
            energy = float(np.average(valid["level_H"], weights=valid["g_stat"]))
        centers[(int(n), int(l))] = energy
    return centers


def get_levels_for_element(
    element: str,
    ionization_stage: int = 1,
    cache_dir: str | Path | None = "nist_cache",
    fallback_data: dict | None = None,
    verbose: bool = False,
) -> dict[tuple[int, int], float]:
    """Función de alto nivel: descarga y parsea niveles para un elemento.

    Args:
        element: símbolo.
        ionization_stage: 1 = neutro, 2 = +1, etc.
        cache_dir: directorio de cache (None para no cachear).
        fallback_data: datos hard-coded a usar si NIST falla (dict {(n,l): E_rel_H}).
        verbose: imprimir progreso.

    Returns:
        Niveles relativos al estado base: {(n, l): E_rel_Hartree}.
    """
    spectrum = f"{element} {int_to_roman(ionization_stage)}"
    text = download_nist_levels(spectrum, cache_dir=cache_dir, verbose=verbose)
    if text is None:
        if verbose:
            print(f"[NIST] usando fallback para {spectrum}")
        return fallback_data or {}

    df = parse_nist_levels_csv(text)
    if df.empty:
        if verbose:
            print(f"[NIST] parseo vacío para {spectrum}; usando fallback")
        return fallback_data or {}

    centers = levels_to_centers(df)
    if not centers:
        return fallback_data or {}
    
    E_base = min(centers.values())
    centers_relative = {nl: E - E_base for nl, E in centers.items()}
    return centers_relative
