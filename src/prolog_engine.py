"""Interfaccia Python verso la knowledge base Prolog (knowledge_base/kb.pl).

SWI-Prolog è interrogato tramite pyswip. Le regole restano in kb.pl; questo modulo
carica i fatti di uno scenario (incidenti, mezzi, registro territoriale, velocità
stimata), interroga la KB e restituisce risultati Python. Non contiene regole di dominio.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
from pyswip import Prolog  # noqa: E402

KB_PATH = config.KB_DIR / "kb.pl"

# Identificatori degli scenari: atomi Prolog validi, senza spazi o caratteri speciali.
_ID = re.compile(r"^[a-z][a-z0-9_]*$")

# Predicati dinamici svuotati prima di ogni scenario.
_DINAMICI = [
    "incidente(_,_,_,_,_,_)",
    "registro(_,_,_)",
    "adiacenza(_,_)",
    "mezzo(_,_)",
    "velocita_stimata(_,_)",
    "assegnato(_,_)",
]


def _valore(x) -> str:
    """Valore di un enum di domain.py, oppure stringa già in forma di atomo."""
    return str(x.value) if hasattr(x, "value") else str(x)


def _id(x) -> str:
    s = str(x)
    if not _ID.match(s):
        raise ValueError(f"identificatore non valido per Prolog: {s!r}")
    return s


class KnowledgeBase:
    """Knowledge base caricata una volta; i fatti di scenario si sostituiscono con reset()."""

    def __init__(self, percorso: Path = KB_PATH):
        self._p = Prolog()
        self._p.consult(str(percorso))

    def _q(self, goal: str) -> list[dict]:
        return list(self._p.query(goal))

    def reset(self) -> None:
        """Rimuove tutti i fatti dello scenario precedente."""
        self._q(", ".join(f"retractall({d})" for d in _DINAMICI))

    def carica_scenario(self, incidenti: list[dict], mezzi: dict, registro: dict,
                        adiacenze: list[tuple], velocita: dict) -> None:
        """Carica i fatti di uno scenario.

        incidenti: dict con chiavi id, tipo, piano, sostanze, persone, meteo.
        mezzi: {id_mezzo: TipoMezzo}.
        registro: {id_incidente: (Vulnerabilita, n_adiacenti)}.
        adiacenze: coppie (id_a, id_b). Vengono asserite in entrambi i versi.
        velocita: {id_incidente: VelocitaFuoco}, velocità vera o stimata.
        """
        self.reset()
        for inc in incidenti:
            self._q(
                f"assertz(incidente({_id(inc['id'])}, {_valore(inc['tipo'])}, {int(inc['piano'])}, "
                f"{_valore(inc['sostanze'])}, {_valore(inc['persone'])}, {_valore(inc['meteo'])}))"
            )
        for m, tipo in mezzi.items():
            self._q(f"assertz(mezzo({_id(m)}, {_valore(tipo)}))")
        for i, (vuln, n_adj) in registro.items():
            self._q(f"assertz(registro({_id(i)}, {_valore(vuln)}, {int(n_adj)}))")
        for a, b in adiacenze:
            self._q(f"assertz(adiacenza({_id(a)}, {_id(b)})), assertz(adiacenza({_id(b)}, {_id(a)}))")
        for i, v in velocita.items():
            self._q(f"assertz(velocita_stimata({_id(i)}, {_valore(v)}))")

    # --- interrogazioni -------------------------------------------------------

    def mezzi_idonei(self, incidente: str) -> list[str]:
        r = self._q(f"mezzi_idonei({_id(incidente)}, L)")
        return sorted(r[0]["L"]) if r else []

    def gravita(self, incidente: str) -> str | None:
        r = self._q(f"gravita({_id(incidente)}, G)")
        return str(r[0]["G"]) if r else None

    def gravita_operativa(self, incidente: str) -> str | None:
        r = self._q(f"gravita_operativa({_id(incidente)}, G)")
        return str(r[0]["G"]) if r else None

    def spiegazioni_velocita(self, incidente: str, gravita: str) -> list[str]:
        """Abduzione: le velocità del fuoco che spiegano la gravità osservata."""
        r = self._q(f"spiegazioni_velocita({_id(incidente)}, {_valore(gravita)}, L)")
        return list(r[0]["L"]) if r else []

    def esposto(self, incidente: str) -> bool:
        return bool(self._q(f"esposto({_id(incidente)})"))

    def collegati(self, incidente: str) -> set[str]:
        """Incidenti raggiungibili per adiacenza (ricorsione su collegato/2), escluso l'incidente stesso."""
        r = self._q(f"setof(B, collegato({_id(incidente)}, B), L) -> true ; L = []")
        return (set(r[0]["L"]) if r else set()) - {incidente}

    def violazioni(self, assegnazioni: list[tuple]) -> list[str]:
        """Verifica una proposta di dispatch come coppie (id_mezzo, id_incidente).

        Le coppie sono una lista, non un dict, così può essere violata anche la regola
        di esclusività (un mezzo assegnato a due incidenti). Lista vuota = proposta valida.
        """
        self._q("retractall(assegnato(_,_))")
        for m, i in assegnazioni:
            self._q(f"assertz(assegnato({_id(m)}, {_id(i)}))")
        r = self._q("findall(S, (violazione(V), format(atom(S), '~w', [V])), L)")
        self._q("retractall(assegnato(_,_))")
        return sorted(r[0]["L"]) if r else []

    def complessita(self) -> dict:
        """Numero di predicati, regole (clausole con corpo) e clausole della KB."""
        r = self._q("kb_statistiche(P, R, C)")
        return {"predicati": int(r[0]["P"]), "regole": int(r[0]["R"]), "clausole": int(r[0]["C"])}


if __name__ == "__main__":
    kb = KnowledgeBase()
    print("KB caricata:", kb.complessita())
