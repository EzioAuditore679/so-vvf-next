"""Estrazione dei campi di una chiamata dal solo testo, con regole e regex.

Adattatore di ingresso, non oggetto di valutazione. Il lessico è definito qui e non
riusa le frasi di PAROLE_CHIAVE: il generatore e l'estrattore non condividono le stesse
espressioni. Il gold set è stato usato anche per analizzare gli errori, quindi il suo
risultato non è completamente indipendente.
"""

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from domain import Meteo, Persone, SostanzePericolose, TipoEdificio

ORDINALI = {"primo": 1, "secondo": 2, "terzo": 3, "quarto": 4, "quinto": 5, "sesto": 6}

# Radici, non frasi intere.
RADICI_EDIFICIO = {
    TipoEdificio.RESIDENZIALE: ["palazzin", "palazz", "condomin", "appartament", "abitazion", "villett", "casa"],
    TipoEdificio.INDUSTRIALE: ["capannon", "stabiliment", "deposit", "officin", "magazzin", "fabbric"],
    TipoEdificio.BOSCO: ["bosc", "sterpaglie", "pineta"],
}

RADICI_SOSTANZE = [
    "bombol", "serbatoi", "vernic", "solvent", "acid", "gas",
    "benzin", "tanic", "gpl", "idrogen", "ossigen", "cloro", "propan",
]

NEGAZIONE = re.compile(r"\b(nessun\w*|niente|nulla)\b|\bnon (c'è|ci sono|ci sia|ci siano|risulta)\b")

# Ordine di priorità: l'incertezza prevale, poi le assenze, poi le conferme.
PATTERN_INCERTE = [r"non sappiamo", r"non si sa", r"forse", r"potrebbe esserci", r"pare ci sia", r"non è sicuro"]
PATTERN_ASSENTI = [
    r"non c'è nessuno",
    r"nessuno dentro",
    r"tutti (sono )?(usciti|fuori)",
    r"edificio (è|risulta) (vuoto|sgombro)",
]
PATTERN_NEGATE_CONFERMATE = re.compile(r"\bnessun\w* (è |sono )?(intrappolat|bloccat|rimast)")
PATTERN_CONFERMATE = [r"intrappolat", r"bloccat", r"rimast\w* dentro", r"non riesc\w* a (uscire|scendere)"]

# La calma va riconosciuta prima del vento, altrimenti "non tira vento" sarebbe vento forte.
PATTERN_CALMO = re.compile(
    r"aria (calma|ferma)|tempo (stabile|calmo)|calm|non (c'è|tira|ci sono|ci sia) vento|niente vento"
)


def _normalizza(testo: str) -> str:
    t = testo.lower().replace("’", "'")
    t = re.sub(r"\bp\.", "piano", t)
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
    if PATTERN_NEGATE_CONFERMATE.search(t):
        return Persone.ASSENTI.value
    if any(re.search(p, t) for p in PATTERN_ASSENTI):
        return Persone.ASSENTI.value
    if any(re.search(p, t) for p in PATTERN_CONFERMATE):
        return Persone.CONFERMATE.value
    return None


def _meteo(t: str) -> str | None:
    if PATTERN_CALMO.search(t):
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
    # ATTENZIONE: numero ottimistico, le regole sono state scritte guardando testi dello stesso tipo.
    # La valutazione vera si fa su gold/test in validate.py.
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
