"""Estrazione dei campi di una chiamata dal solo testo (regole e regex).

Il lessico di questo modulo è definito qui, separato da PAROLE_CHIAVE di
domain.py che il generatore usa per produrre i testi. Così il risultato non
dipende dalle stesse frasi usate per generare il dataset.

"""

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from domain import Meteo, Persone, SostanzePericolose, TipoEdificio  # noqa: E402

ORDINALI = {"primo": 1, "secondo": 2, "terzo": 3, "quarto": 4, "quinto": 5, "sesto": 6}

# Radici (non frasi intere) per riconoscere il tipo di edificio.
RADICI_EDIFICIO = {
    TipoEdificio.RESIDENZIALE: ["palazzin", "condomin", "appartament", "abitazion"],
    TipoEdificio.INDUSTRIALE: ["capannon", "stabiliment", "deposit"],
    TipoEdificio.BOSCO: ["bosc", "sterpaglie", "pineta"],
}

# Radici che indicano una sostanza pericolosa.
RADICI_SOSTANZE = ["bombol", "serbatoi", "vernic", "solvent", "acid", "gas"]

NEGAZIONE = re.compile(r"\b(nessun\w*|niente|nulla)\b")

# Pattern per le persone, in ordine di priorità: l'incertezza prevale sulle affermazioni.
PATTERN_INCERTE = [r"non sappiamo", r"forse", r"potrebbe esserci"]
PATTERN_ASSENTI = [r"non c'è nessuno", r"tutti sono usciti", r"edificio è vuoto"]
PATTERN_CONFERMATE = [r"intrappolat", r"bloccat", r"non riesc\w* a uscire"]


def _normalizza(testo: str) -> str:
    t = testo.lower().replace("’", "'")
    t = re.sub(r"\bp\.", "piano", t)  # abbreviazione del parlato
    return re.sub(r"\s+", " ", t)


def _frasi(t: str) -> list[str]:
    return [f for f in re.split(r"[.!?]+", t) if f.strip()]


def _piano(t: str) -> int:
    """Piano come intero. Restituisce 0 se non è indicato (es. bosco)."""
    m = re.search(r"\b(\d+|" + "|".join(ORDINALI) + r")\s*(?:°\s*)?piano", t)
    if not m:
        return 0
    v = m.group(1)
    return int(v) if v.isdigit() else ORDINALI[v]


def _edificio(t: str) -> str | None:
    for tipo, radici in RADICI_EDIFICIO.items():
        if any(r in t for r in radici):
            return tipo.value
    return None


def _sostanze(t: str) -> str:
    """Una frase positiva è 'si' solo se non è negata. Il default è 'no'."""
    for f in _frasi(t):
        contiene_sostanza = any(r in f for r in RADICI_SOSTANZE) or "sostanz" in f or "rischio chimico" in f
        if not contiene_sostanza:
            continue
        if NEGAZIONE.search(f):
            return SostanzePericolose.NO.value
        if any(r in f for r in RADICI_SOSTANZE):
            return SostanzePericolose.SI.value
    return SostanzePericolose.NO.value


def _persone(t: str) -> str | None:
    if any(re.search(p, t) for p in PATTERN_INCERTE):
        return Persone.INCERTE.value
    if any(re.search(p, t) for p in PATTERN_ASSENTI):
        return Persone.ASSENTI.value
    if any(re.search(p, t) for p in PATTERN_CONFERMATE):
        return Persone.CONFERMATE.value
    return None


def _meteo(t: str) -> str | None:
    if re.search(r"non c'è vento|calm", t):
        return Meteo.CALMO.value
    if "vento" in t:
        return Meteo.VENTO_FORTE.value
    return None


def estrai_campi(testo: str) -> dict:
    """Ricava tutti i campi dal testo. I campi non riconosciuti restano None."""
    t = _normalizza(testo)
    return {
        "tipo_edificio": _edificio(t),
        "piano": _piano(t),
        "sostanze_pericolose": _sostanze(t),
        "persone": _persone(t),
        "meteo": _meteo(t),
    }


if __name__ == "__main__":
    # Verifica rapida sul dataset sintetico. ATTENZIONE: questo numero è ottimistico,
    # perché le regole sono state scritte guardando lo stesso tipo di testi.
    # La valutazione vera si fa su gold set in validate.py.
    percorso = ROOT / "data" / "dataset_sintetico.csv"
    campi = ["tipo_edificio", "piano", "sostanze_pericolose", "persone", "meteo"]
    corretti = {c: 0 for c in campi}
    righe = 0
    with percorso.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            righe += 1
            pred = estrai_campi(r["testo"])
            for c in campi:
                if str(pred[c]) == str(r[c]) or (c == "piano" and int(r[c]) == pred[c]):
                    corretti[c] += 1
    for c in campi:
        print(f"{c}: {corretti[c] / righe:.3f}")
