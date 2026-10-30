"""Genera knowledge_base/kb_statica.pl dalle costanti Python.

Soglie e punteggi della regola di gravità e la soglia dell'autoscala sono definiti
una sola volta in Python. Questo modulo li riscrive in Prolog, così kb.pl non duplica
valori che potrebbero divergere.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config
import modello_generativo as mg
from domain import SOGLIA_PIANI_AUTOSCALA

KB_STATICA_PATH = config.KB_DIR / "kb_statica.pl"


def genera_kb_statica() -> str:
    righe = ["% Generato da src/prolog_export.py. Non modificare a mano."]
    righe.append(f"soglia_piani_autoscala({SOGLIA_PIANI_AUTOSCALA}).")
    righe += [f"punti_persone({p.value}, {n})." for p, n in mg.PUNTI_PERSONE.items()]
    righe += [f"punti_velocita({v.value}, {n})." for v, n in mg.PUNTI_VELOCITA.items()]
    righe.append(f"soglia_critico({mg.SOGLIA_CRITICO}).")
    righe.append(f"soglia_alto({mg.SOGLIA_ALTO}).")
    return "\n".join(righe) + "\n"


def salva(percorso: Path = KB_STATICA_PATH) -> None:
    percorso.parent.mkdir(parents=True, exist_ok=True)
    percorso.write_text(genera_kb_statica(), encoding="utf-8")


if __name__ == "__main__":
    salva()
    print(f"Scritto {KB_STATICA_PATH}")
