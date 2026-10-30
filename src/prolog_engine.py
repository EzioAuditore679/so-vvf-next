"""Interfaccia Python verso la knowledge base Prolog (SWI-Prolog tramite pyswip).

Rigenera la parte statica della KB, poi carica i fatti di uno scenario e interroga
le regole di kb.pl. Non contiene regole di dominio.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config  
import prolog_export  
from pyswip import Prolog  

KB_PATH = config.KB_DIR / "kb.pl"
_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_DINAMICI = [
    "incidente(_,_,_,_,_,_)",
    "registro(_,_,_)",
    "adiacenza(_,_)",
    "mezzo(_,_)",
    "velocita_stimata(_,_)",
    "gravita_stimata(_,_)",
    "assegnato(_,_,_)",
]
RUOLI = ["estinzione", "rinforzo", "soccorso_quota", "decontaminazione"]


def _valore(x) -> str:
    return str(x.value) if hasattr(x, "value") else str(x)


def _id(x) -> str:
    s = str(x)
    if not _ID.match(s):
        raise ValueError(f"identificatore non valido per Prolog: {s!r}")
    return s


class KnowledgeBase:

    def __init__(self, percorso: Path = KB_PATH):
        prolog_export.salva()
        self._p = Prolog()
        self._p.consult(str(percorso))

    def _q(self, goal: str) -> list[dict]:
        return list(self._p.query(goal))

    def reset(self) -> None:
        self._q(", ".join(f"retractall({d})" for d in _DINAMICI))

    def carica_scenario(self, incidenti: list[dict], mezzi: dict, registro: dict,
                        adiacenze: list[tuple], velocita: dict | None = None,
                        gravita_stimata: dict | None = None) -> None:
        """Carica i fatti di uno scenario.

        incidenti: dict con chiavi id, tipo, piano, sostanze, persone, meteo.
        mezzi: {id_mezzo: TipoMezzo}. registro: {id_incidente: (Vulnerabilita, n_adiacenti)}.
        adiacenze: coppie (id_a, id_b). velocita: {id_incidente: VelocitaFuoco}.
        gravita_stimata: {id_incidente: Gravita} da una fonte esterna (ML, rete). Se presente,
        prevale sulla regola applicata alla velocità (gravita_base/2 in kb.pl).
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
        for i, v in (velocita or {}).items():
            self._q(f"assertz(velocita_stimata({_id(i)}, {_valore(v)}))")
        for i, g in (gravita_stimata or {}).items():
            self._q(f"assertz(gravita_stimata({_id(i)}, {_valore(g)}))")

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
        r = self._q(f"spiegazioni_velocita({_id(incidente)}, {_valore(gravita)}, L)")
        return list(r[0]["L"]) if r else []

    def esposto(self, incidente: str) -> bool:
        return bool(self._q(f"esposto({_id(incidente)})"))

    def collegati(self, incidente: str) -> set[str]:
        """Incidenti raggiungibili per adiacenza, escluso l'incidente stesso."""
        r = self._q(f"setof(B, collegato({_id(incidente)}, B), L) -> true ; L = []")
        return (set(r[0]["L"]) if r else set()) - {incidente}

    def squadra(self, incidente: str) -> list[str]:
        """Ruoli richiesti dall'incidente, nell'ordine di RUOLI."""
        r = self._q(f"squadra_richiesta({_id(incidente)}, L)")
        ruoli = {str(x) for x in r[0]["L"]} if r else set()
        return [x for x in RUOLI if x in ruoli]

    def mezzi_adatti(self, incidente: str, ruolo: str) -> list[str]:
        r = self._q(f"mezzi_adatti({_id(incidente)}, {_id(ruolo)}, L)")
        return sorted(str(m) for m in r[0]["L"]) if r else []

    def motivi(self, incidente: str) -> list[tuple[str, str]]:
        """Spiegazione della squadra: coppie (ruolo, motivo) prodotte da motivo/3."""
        r = self._q(f"findall(S, (motivo({_id(incidente)}, R, M), format(atom(S), '~w|~w', [R, M])), L)")
        coppie = []
        for x in (r[0]["L"] if r else []):
            ruolo, motivo = str(x).split("|", 1)
            coppie.append((ruolo, motivo))
        return sorted(set(coppie), key=lambda c: (RUOLI.index(c[0]), c[1]))

    def _con_assegnazioni(self, assegnazioni: list[tuple], goal: str) -> list[str]:
        self._q("retractall(assegnato(_,_,_))")
        for m, i, ruolo in assegnazioni:
            self._q(f"assertz(assegnato({_id(m)}, {_id(i)}, {_id(ruolo)}))")
        r = self._q(goal)
        self._q("retractall(assegnato(_,_,_))")
        return sorted(str(x) for x in r[0]["L"]) if r else []

    def violazioni(self, assegnazioni: list[tuple]) -> list[str]:
        """Verifica una proposta di dispatch come triple (id_mezzo, id_incidente, ruolo).

        Una lista, non un dict, permette di esprimere anche un mezzo assegnato due volte.
        Lista vuota = proposta valida. I ruoli scoperti sono riportati da ruoli_scoperti.
        """
        return self._con_assegnazioni(
            assegnazioni, "findall(S, (violazione(V), format(atom(S), '~w', [V])), L)")

    def ruoli_scoperti(self, assegnazioni: list[tuple]) -> list[str]:
        return self._con_assegnazioni(
            assegnazioni, "findall(S, (ruolo_scoperto(I, R), format(atom(S), '~w-~w', [I, R])), L)")

    def complessita(self) -> dict:
        r = self._q("kb_statistiche(P, R, C)")
        return {"predicati": int(r[0]["P"]), "regole": int(r[0]["R"]), "clausole": int(r[0]["C"])}


if __name__ == "__main__":
    print("KB caricata:", KnowledgeBase().complessita())
