"""Ricerca nello spazio degli stati: distanze sulla griglia e dispatch dei mezzi.

Due livelli, con dimostrazioni in docs/ricerca.md:
- livello 1: distanza tra una stazione e un incidente sulla griglia con ostacoli (A*, Manhattan);
- livello 2: assegnazione degli incidenti ai mezzi, con costi dal livello 1 (LcFS, A* con h_min o h_libero).

Il modulo non dipende da Prolog: gli idonei sono forniti dal chiamante (vedi idonei_con_kb).
"""

from __future__ import annotations

import csv
import heapq
import itertools
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
import modello_generativo as mg  # noqa: E402
from domain import TipoMezzo  # noqa: E402

INF = math.inf
COSTO_MINIMO = 1  # epsilon della Proposizione 2.1: ogni assegnazione costa almeno 1


@dataclass(frozen=True)
class Risultato:
    costo: float
    cammino: tuple | None
    espansi: int
    generati: int
    completo: bool  # False se la ricerca ha superato config.LIMITE_NODI


def _cammino(padre: dict, s) -> tuple:
    cammino = []
    while s is not None:
        cammino.append(s)
        s = padre[s]
    return tuple(reversed(cammino))


def ricerca(inizio, obiettivo: Callable, successori: Callable, euristica: Callable,
            limite: int = config.LIMITE_NODI) -> Risultato:
    """A* su grafo con lista dei chiusi. Con euristica nulla è LcFS (ricerca a costo uniforme).

    Il test di obiettivo è fatto alla selezione dalla frontiera. Con euristica consistente
    un nodo chiuso non viene mai riaperto, quindi il primo obiettivo estratto è ottimo.
    """
    conta = itertools.count()
    frontiera = [(euristica(inizio), 0, next(conta), inizio)]
    g_migliore = {inizio: 0}
    padre = {inizio: None}
    chiusi = set()
    espansi = 0
    generati = 1
    while frontiera:
        _, g, _, s = heapq.heappop(frontiera)
        if s in chiusi:
            continue
        chiusi.add(s)
        if obiettivo(s):
            return Risultato(g, _cammino(padre, s), espansi, generati, True)
        if espansi >= limite:
            return Risultato(INF, None, espansi, generati, False)
        espansi += 1
        for s2, costo_arco in successori(s):
            g2 = g + costo_arco
            if g2 >= g_migliore.get(s2, INF):
                continue
            h2 = euristica(s2)
            if h2 == INF:
                continue
            g_migliore[s2] = g2
            padre[s2] = s
            generati += 1
            heapq.heappush(frontiera, (g2 + h2, g2, next(conta), s2))
    return Risultato(INF, None, espansi, generati, True)


# --- Livello 1: distanze sulla griglia ---------------------------------------

Cella = tuple[int, int]


@dataclass(frozen=True)
class Griglia:
    righe: int
    colonne: int
    ostacoli: frozenset

    def libera(self, c: Cella) -> bool:
        r, k = c
        return 0 <= r < self.righe and 0 <= k < self.colonne and c not in self.ostacoli

    def successori(self, c: Cella):
        r, k = c
        for dr, dk in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            v = (r + dr, k + dk)
            if self.libera(v):
                yield v, 1


def manhattan(a: Cella, b: Cella) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def distanza_griglia(g: Griglia, partenza: Cella, arrivo: Cella, con_euristica: bool = True) -> Risultato:
    h = (lambda c: manhattan(c, arrivo)) if con_euristica else (lambda c: 0)
    return ricerca(partenza, lambda c: c == arrivo, g.successori, h)


def genera_griglia(rng: random.Random, righe: int, colonne: int, p_ostacolo: float) -> Griglia:
    ostacoli = frozenset(
        (r, k) for r in range(righe) for k in range(colonne) if rng.random() < p_ostacolo
    )
    return Griglia(righe, colonne, ostacoli)


# --- Livello 2: dispatch -----------------------------------------------------

@dataclass
class Scenario:
    incidenti: tuple
    mezzi: tuple
    idonei: dict        # incidente -> lista di mezzi idonei (dalla KB Prolog)
    costo: dict         # (mezzo, incidente) -> max(distanza, COSTO_MINIMO), INF se irraggiungibile


@dataclass
class Caso:
    griglia: Griglia
    mezzi: dict         # id -> (TipoMezzo, cella)
    incidenti: dict     # id -> (variabili di modello_generativo, cella)


def _servite(s: frozenset) -> set:
    return {i for _, i in s}


def _occupati(s: frozenset) -> set:
    return {m for m, _ in s}


class Dispatch:
    """Stati = insiemi di assegnazioni (mezzo, incidente). Costo di un'azione = costo[(m, i)].

    Le euristiche sono ammissibili e consistenti: dimostrazioni in docs/ricerca.md.
    """

    def __init__(self, sc: Scenario):
        self.sc = sc
        self.ammessi = {
            i: [m for m in sc.idonei.get(i, []) if sc.costo[(m, i)] < INF] for i in sc.incidenti
        }
        self.minimo = {
            i: min((sc.costo[(m, i)] for m in self.ammessi[i]), default=INF) for i in sc.incidenti
        }

    def inizio(self) -> frozenset:
        return frozenset()

    def obiettivo(self, s: frozenset) -> bool:
        return len(_servite(s)) == len(self.sc.incidenti)

    def successori(self, s: frozenset):
        servite = _servite(s)
        occupati = _occupati(s)
        for i in self.sc.incidenti:
            if i in servite:
                continue
            for m in self.ammessi[i]:
                if m not in occupati:
                    yield s | {(m, i)}, self.sc.costo[(m, i)]

    def h_min(self, s: frozenset) -> float:
        """Rilassamento: si rimuove l'esclusività dei mezzi. Minimo statico per incidente."""
        servite = _servite(s)
        return sum(self.minimo[i] for i in self.sc.incidenti if i not in servite)

    def h_libero(self, s: frozenset) -> float:
        """Rilassamento: esclusività solo tra incidenti residui. Ammessi solo i mezzi liberi."""
        servite = _servite(s)
        occupati = _occupati(s)
        totale = 0
        for i in self.sc.incidenti:
            if i in servite:
                continue
            liberi = [self.sc.costo[(m, i)] for m in self.ammessi[i] if m not in occupati]
            if not liberi:
                return INF
            totale += min(liberi)
        return totale


def _scenario_con_costi(caso: Caso, idonei: dict) -> Scenario | None:
    costo = {}
    for i, (_, cella_i) in caso.incidenti.items():
        for m in idonei.get(i, []):
            _, cella_m = caso.mezzi[m]
            r = distanza_griglia(caso.griglia, cella_m, cella_i)
            costo[(m, i)] = max(r.costo, COSTO_MINIMO) if r.completo and r.costo < INF else INF
    for i in caso.incidenti:
        if not any(costo.get((m, i), INF) < INF for m in idonei.get(i, [])):
            return None
    return Scenario(tuple(caso.incidenti), tuple(caso.mezzi), idonei, costo)


def genera_caso(rng: random.Random, n_incidenti: int, n_mezzi: int) -> Caso:
    griglia = genera_griglia(rng, *config.DIM_GRIGLIA, config.PROB_OSTACOLO)
    liberi = [(r, k) for r in range(griglia.righe) for k in range(griglia.colonne)
              if (r, k) not in griglia.ostacoli]
    mezzi = {f"m{k}": (rng.choice(list(TipoMezzo)), rng.choice(liberi)) for k in range(n_mezzi)}
    incidenti = {f"i{k}": (mg.genera_variabili(rng), rng.choice(liberi)) for k in range(n_incidenti)}
    return Caso(griglia, mezzi, incidenti)


def prepara_scenario(caso: Caso, idonei_fn: Callable) -> Scenario | None:
    """Calcola gli idonei e i costi del livello 2. None se un incidente non è raggiungibile."""
    return _scenario_con_costi(caso, idonei_fn(caso))


def idonei_con_kb(kb) -> Callable:
    """Restituisce la funzione che ricava gli idonei dalla KB Prolog per un caso."""
    def idonei(caso: Caso) -> dict:
        fatti = {i: v for i, (v, _) in caso.incidenti.items()}
        kb.carica_scenario(
            [dict(id=i, tipo=v["tipo_edificio"], piano=v["piano"], sostanze=v["sostanze_pericolose"],
                  persone=v["persone"], meteo=v["meteo"]) for i, v in fatti.items()],
            {m: t for m, (t, _) in caso.mezzi.items()},
            {i: (v["vulnerabilita"], v["adiacenti"]) for i, v in fatti.items()},
            [],
            {i: v["velocita_fuoco"] for i, v in fatti.items()},
        )
        return {i: kb.mezzi_idonei(i) for i in caso.incidenti}
    return idonei


# --- Confronto sperimentale --------------------------------------------------

ALGORITMI = ["LcFS", "A* h_min", "A* h_libero"]


def _esegui(p: Dispatch, algoritmo: str) -> Risultato:
    euristica = {
        "LcFS": lambda s: 0,
        "A* h_min": p.h_min,
        "A* h_libero": p.h_libero,
    }[algoritmo]
    return ricerca(p.inizio(), p.obiettivo, p.successori, euristica)


def _dev(valori: list) -> float:
    return statistics.stdev(valori) if len(valori) > 1 else 0.0


def confronto(seme: int, dimensioni: list, n_scenari: int, idonei_fn: Callable,
              mezzi_per_incidente: float = config.N_MEZZI_PER_INCIDENTE) -> list[dict]:
    rng = random.Random(seme)
    righe = []
    for n in dimensioni:
        n_mezzi = math.ceil(n * mezzi_per_incidente)
        valide = 0
        tentativi = 0
        while valide < n_scenari:
            tentativi += 1
            if tentativi > 1000 * n_scenari:
                raise RuntimeError(f"troppi casi senza soluzione per n={n}")
            sc = prepara_scenario(genera_caso(rng, n, n_mezzi), idonei_fn)
            if sc is None:
                continue
            valide += 1
            p = Dispatch(sc)
            esiti = {}
            for alg in ALGORITMI:
                t0 = time.perf_counter()
                esiti[alg] = (_esegui(p, alg), time.perf_counter() - t0)
            costi = {r.costo for r, _ in esiti.values() if r.completo}
            if len(costi) > 1:
                raise AssertionError(f"algoritmi completi con costi diversi: {costi}")
            for alg, (r, secondi) in esiti.items():
                righe.append(dict(dimensione=n, scenario=valide, algoritmo=alg, costo=r.costo,
                                  espansi=r.espansi, generati=r.generati, secondi=secondi,
                                  completo=r.completo))
    return righe


def riepilogo(righe: list[dict]) -> list[dict]:
    gruppi: dict = {}
    for r in righe:
        gruppi.setdefault((r["dimensione"], r["algoritmo"]), []).append(r)
    out = []
    for (n, alg), rs in sorted(gruppi.items()):
        esp = [r["espansi"] for r in rs]
        sec = [r["secondi"] for r in rs]
        out.append(dict(
            dimensione=n, algoritmo=alg, casi=len(rs),
            completi=sum(r["completo"] for r in rs),
            espansi_media=statistics.mean(esp), espansi_dev=_dev(esp),
            secondi_media=statistics.mean(sec), secondi_dev=_dev(sec),
        ))
    return out


def _scrivi_csv(righe: list[dict], percorso: Path) -> None:
    percorso.parent.mkdir(parents=True, exist_ok=True)
    with percorso.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(righe[0].keys()))
        writer.writeheader()
        writer.writerows(righe)


if __name__ == "__main__":
    from prolog_engine import KnowledgeBase

    righe = confronto(config.SEME, config.N_INCIDENTI_SCENARI, config.N_SCENARI_PER_DIMENSIONE,
                      idonei_con_kb(KnowledgeBase()))
    _scrivi_csv(righe, config.RESULTS_DIR / "search_confronto.csv")
    tabella = riepilogo(righe)
    _scrivi_csv(tabella, config.RESULTS_DIR / "search_riepilogo.csv")
    print(f"{'n':>3} {'algoritmo':12s} {'completi':>9s} {'espansi (media ± dev)':>24s} {'secondi (media)':>16s}")
    for t in tabella:
        print(f"{t['dimensione']:>3} {t['algoritmo']:12s} {t['completi']:>4}/{t['casi']:<4} "
              f"{t['espansi_media']:>12.1f} ± {t['espansi_dev']:<9.1f} {t['secondi_media']:>16.6f}")
