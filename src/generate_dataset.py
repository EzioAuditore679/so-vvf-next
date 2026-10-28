"""Generatore di chiamate al 115 sintetiche con valori veri dei campi.

Ogni record contiene un testo in italiano parlato (con rumore: minuscole,
riempitivi, abbreviazioni, punteggiatura assente) e i valori veri dei campi.
nlp_extract.py dovrà ricavare i campi dal solo testo; i valori veri servono
solo a validate.py per misurare l'accuratezza dell'estrazione.

"""

import csv
import random
import re
import sys
from collections import Counter
from pathlib import Path
import config 

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from domain import (  
    PAROLE_CHIAVE,
    Campo,
    Meteo,
    Persone,
    SostanzePericolose,
    TipoEdificio,
)

ORDINALI = {1: "primo", 2: "secondo", 3: "terzo", 4: "quarto", 5: "quinto", 6: "sesto"}

APERTURE = [
    "Pronto, c'è",
    "Salve, abbiamo",
    "Qui c'è",
    "Sono sul posto, c'è",
    "Segnalo",
    "Buongiorno, c'è",
]

# Frasi per il meteo: il valore vero è in Meteo, la frase è solo il modo di dirlo.
FRASI_METEO = {
    Meteo.VENTO_FORTE: ["c'è vento forte", "il vento è forte"],
    Meteo.CALMO: ["non c'è vento", "il tempo è calmo"],
}

RIEMPITIVI = ["eh", "ehm", "cioè", "allora"]


def _frase_edificio(rng: random.Random, edificio: TipoEdificio, piano: int) -> str:
    kw = rng.choice(PAROLE_CHIAVE[Campo.TIPO_EDIFICIO][edificio])
    if edificio is TipoEdificio.BOSCO:
        return f"un incendio in {kw}"
    return f"un incendio in {kw} al {ORDINALI[piano]} piano"


def _frase_sostanze(rng: random.Random, sostanze: SostanzePericolose) -> str:
    kw = rng.choice(PAROLE_CHIAVE[Campo.SOSTANZE_PERICOLOSE][sostanze])
    if sostanze is SostanzePericolose.SI:
        return f"vicino alle fiamme ci sono {kw}"
    return kw


def _frase_persone(rng: random.Random, persone: Persone) -> str:
    kw = rng.choice(PAROLE_CHIAVE[Campo.PERSONE][persone])
    return f"mi dicono che {kw}"


def _rumore(rng: random.Random, testo: str) -> str:
    """Applica le imperfezioni tipiche del parlato. Ogni trasformazione ha una probabilità fissa."""
    if rng.random() < 0.3:
        testo = testo.lower()
    if rng.random() < 0.3:
        m = re.search(r"\. (\w)", testo)
        if m:
            riempitivo = rng.choice(RIEMPITIVI)
            testo = testo[: m.start()] + f" {riempitivo}, " + m.group(1).lower() + testo[m.end():]
    if rng.random() < 0.2:
        testo = testo.replace("piano", "p.")
    if rng.random() < 0.3:
        testo = testo.rstrip(".")
    return testo


def genera_record(rng: random.Random, idx: int) -> dict:
    edificio = rng.choice(list(TipoEdificio))
    piano = 0 if edificio is TipoEdificio.BOSCO else rng.randint(1, 6)
    sostanze = rng.choice(list(SostanzePericolose))
    persone = rng.choice(list(Persone))
    meteo = rng.choice(list(Meteo))

    apertura = rng.choice(APERTURE)
    frasi = [
        f"{apertura} {_frase_edificio(rng, edificio, piano)}",
        _frase_sostanze(rng, sostanze).capitalize(),
        _frase_persone(rng, persone).capitalize(),
        rng.choice(FRASI_METEO[meteo]).capitalize(),
    ]
    testo = _rumore(rng, ". ".join(frasi) + ".")

    return {
        "id": f"c{idx:05d}",
        "testo": testo,
        "tipo_edificio": edificio.value,
        "piano": piano,
        "sostanze_pericolose": sostanze.value,
        "persone": persone.value,
        "meteo": meteo.value,
    }


def genera(n: int, seme: int) -> list[dict]:
    rng = random.Random(seme)
    return [genera_record(rng, i) for i in range(n)]


def scrivi_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def copertura(records: list[dict]) -> dict[str, Counter]:
    """Conteggio dei valori per campo: serve a verificare che ogni etichetta compaia nel dataset."""
    campi = ["tipo_edificio", "sostanze_pericolose", "persone", "meteo"]
    return {c: Counter(r[c] for r in records) for c in campi}


if __name__ == "__main__":
    records = genera(config.N_TESTI_SINTETICI, config.SEME)
    out = config.DATA_DIR / "dataset_sintetico.csv"
    scrivi_csv(records, out)
    print(f"Scritti {len(records)} record in {out}")
    for campo, conteggi in copertura(records).items():
        print(f"  {campo}: {dict(conteggi)}")
