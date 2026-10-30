"""Ricerca nello spazio degli stati: distanze sulla griglia e dispatch dei mezzi.

Due livelli:
- livello 1: distanza tra una stazione e un incidente sulla griglia con ostacoli (A*, Manhattan);
- livello 2: copertura dei ruoli richiesti dagli incidenti, con costi dal livello 1 (LcFS, A* con h_min
  o h_libero). I ruoli e i mezzi adatti vengono dalla KB Prolog (adatti_con_kb).
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

import config
import modello_generativo as mg
from domain import TipoMezzo

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


# --- Livello 1: distanze sulla griglia ---
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


# --- Livello 2: dispatch per ruoli ---

@dataclass
class Scenario:
    ruoli: tuple        # ruoli da coprire, nell'ordine di assegnazione (vedi ordina_ruoli)
    mezzi: tuple
    adatti: dict        # (incidente, ruolo) -> mezzi che lo possono coprire (dalla KB Prolog)
    costo: dict         # (mezzo, incidente) -> max(distanza, COSTO_MINIMO), INF se irraggiungibile
    penalita: float = config.PENALITA_RUOLO_SCOPERTO


@dataclass
class Caso:
    griglia: Griglia
    mezzi: dict         # id -> (TipoMezzo, cella)
    incidenti: dict     # id -> (variabili di modello_generativo, cella)


def ordina_ruoli(ruoli: list, ammessi: dict) -> tuple:
    """Ordine fisso delle variabili: prima i ruoli con meno mezzi ammessi (MRV statico dei CSP).

    L'ordine non cambia l'ottimo, perché ogni soluzione resta raggiungibile; riduce i rami.
    """
    return tuple(sorted(ruoli, key=lambda r: (len(ammessi[r]), r[0], r[1])))


class Dispatch:
    """CSP risolto con la ricerca: una variabile per ruolo, valori = mezzi adatti o 'scoperto'.

    Stato = (k, mezzi occupati): i primi k ruoli sono decisi. Due assegnazioni parziali con lo stesso
    k e gli stessi mezzi occupati hanno lo stesso futuro, quindi sono lo stesso stato (potatura
    dei cammini multipli). Azioni sul ruolo k: un mezzo adatto e libero, al costo C(m, i), oppure
    lasciarlo scoperto, al costo della penalità. Euristiche: docs/ricerca.md §2.3–2.4.
    """

    def __init__(self, sc: Scenario):
        self.sc = sc
        self.ammessi = {r: [m for m in sc.adatti.get(r, []) if sc.costo.get((m, r[0]), INF) < INF]
                        for r in sc.ruoli}
        self.mu = [min([sc.penalita] + [sc.costo[(m, r[0])] for m in self.ammessi[r]])
                   for r in sc.ruoli]
        self.coda_mu = [sum(self.mu[k:]) for k in range(len(sc.ruoli) + 1)]

    def inizio(self) -> tuple:
        return (0, frozenset())

    def obiettivo(self, s: tuple) -> bool:
        return s[0] == len(self.sc.ruoli)

    def successori(self, s: tuple):
        k, occupati = s
        if k == len(self.sc.ruoli):
            return
        r = self.sc.ruoli[k]
        for m in self.ammessi[r]:
            if m not in occupati:
                yield (k + 1, occupati | {m}), self.sc.costo[(m, r[0])]
        yield (k + 1, occupati), self.sc.penalita

    def h_min(self, s: tuple) -> float:
        """Rilassamento: si rimuove l'esclusività dei mezzi. Minimo statico per ruolo."""
        return self.coda_mu[s[0]]

    def h_libero(self, s: tuple) -> float:
        """Rilassamento: esclusività solo verso i mezzi già occupati, non tra ruoli residui."""
        k, occupati = s
        totale = 0
        for r in self.sc.ruoli[k:]:
            totale += min([self.sc.penalita] + [self.sc.costo[(m, r[0])] for m in self.ammessi[r]
                                                if m not in occupati])
        return totale

    def piano(self, cammino: tuple) -> tuple[list, list]:
        """Dal cammino di stati: assegnazioni (mezzo, incidente, ruolo) e ruoli scoperti."""
        assegnazioni, scoperti = [], []
        for (k, occ), (_, occ2) in zip(cammino, cammino[1:]):
            i, ruolo = self.sc.ruoli[k]
            nuovo = occ2 - occ
            if nuovo:
                assegnazioni.append((next(iter(nuovo)), i, ruolo))
            else:
                scoperti.append((i, ruolo))
        return assegnazioni, scoperti


def costi_caso(caso: Caso) -> dict:
    """Costo C(m, i) del livello 2 per ogni mezzo e incidente, dal livello 1."""
    costo = {}
    for i, (_, cella_i) in caso.incidenti.items():
        for m, (_, cella_m) in caso.mezzi.items():
            r = distanza_griglia(caso.griglia, cella_m, cella_i)
            costo[(m, i)] = max(r.costo, COSTO_MINIMO) if r.completo and r.costo < INF else INF
    return costo


def scenario_da_adatti(adatti: dict, mezzi: tuple, costo: dict,
                       penalita: float = config.PENALITA_RUOLO_SCOPERTO) -> Scenario:
    ammessi = {r: [m for m in ms if costo.get((m, r[0]), INF) < INF] for r, ms in adatti.items()}
    return Scenario(ordina_ruoli(list(adatti), ammessi), tuple(mezzi), adatti, costo, penalita)


def adiacenze_caso(caso: Caso, soglia: int = config.DISTANZA_ADIACENZA) -> list[tuple]:
    """Coppie di incidenti abbastanza vicini da propagare il pericolo."""
    ids = sorted(caso.incidenti)
    return [(a, b) for x, a in enumerate(ids) for b in ids[x + 1:]
            if manhattan(caso.incidenti[a][1], caso.incidenti[b][1]) <= soglia]


def genera_caso(rng: random.Random, n_incidenti: int, n_mezzi: int,
                variabili: Callable | None = None) -> Caso:
    """Griglia, flotta con la composizione di config e incidenti dal modello generativo."""
    griglia = genera_griglia(rng, *config.DIM_GRIGLIA, config.PROB_OSTACOLO)
    liberi = [(r, k) for r in range(griglia.righe) for k in range(griglia.colonne)
              if (r, k) not in griglia.ostacoli]
    tipi = [TipoMezzo(t) for t in config.COMPOSIZIONE_FLOTTA]
    pesi = list(config.COMPOSIZIONE_FLOTTA.values())
    mezzi = {f"m{k}": (rng.choices(tipi, pesi)[0], rng.choice(liberi)) for k in range(n_mezzi)}
    genera = variabili or mg.genera_variabili
    incidenti = {f"i{k}": (genera(rng), rng.choice(liberi)) for k in range(n_incidenti)}
    return Caso(griglia, mezzi, incidenti)


def carica_caso(kb, caso: Caso, gravita_stimata: dict | None = None) -> None:
    """Asserisce nella KB i fatti del caso. Senza gravita_stimata vale la regola con la velocità vera."""
    v = {i: var for i, (var, _) in caso.incidenti.items()}
    kb.carica_scenario(
        [dict(id=i, tipo=x["tipo_edificio"], piano=x["piano"], sostanze=x["sostanze_pericolose"],
              persone=x["persone"], meteo=x["meteo"]) for i, x in v.items()],
        {m: t for m, (t, _) in caso.mezzi.items()},
        {i: (x["vulnerabilita"], x["adiacenti"]) for i, x in v.items()},
        adiacenze_caso(caso),
        {i: x["velocita_fuoco"] for i, x in v.items()},
        gravita_stimata,
    )


def adatti_con_kb(kb, caso: Caso, gravita_stimata: dict | None = None) -> dict:
    """Ruoli richiesti e mezzi adatti, ricavati dalla KB Prolog."""
    carica_caso(kb, caso, gravita_stimata)
    return {(i, r): kb.mezzi_adatti(i, r) for i in sorted(caso.incidenti) for r in kb.squadra(i)}


def prepara_scenario(caso: Caso, adatti_fn: Callable) -> Scenario:
    """Ruoli e mezzi adatti da adatti_fn(caso), costi dal livello 1."""
    return scenario_da_adatti(adatti_fn(caso), tuple(caso.mezzi), costi_caso(caso))


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


def confronto(seme: int, dimensioni: list, n_scenari: int, adatti_fn: Callable,
              mezzi_per_incidente: float = config.N_MEZZI_PER_INCIDENTE) -> list[dict]:
    rng = random.Random(seme)
    righe = []
    for n in dimensioni:
        n_mezzi = math.ceil(n * mezzi_per_incidente)
        for k in range(1, n_scenari + 1):
            sc = prepara_scenario(genera_caso(rng, n, n_mezzi), adatti_fn)
            p = Dispatch(sc)
            esiti = {}
            for alg in ALGORITMI:
                t0 = time.perf_counter()
                esiti[alg] = (_esegui(p, alg), time.perf_counter() - t0)
            costi = {r.costo for r, _ in esiti.values() if r.completo}
            if len(costi) > 1:
                raise AssertionError(f"algoritmi completi con costi diversi: {costi}")
            migliore = esiti["A* h_libero"][0]
            scoperti = len(p.piano(migliore.cammino)[1]) if migliore.completo else None
            for alg, (r, secondi) in esiti.items():
                righe.append(dict(dimensione=n, scenario=k, ruoli=len(sc.ruoli), algoritmo=alg,
                                  costo=r.costo, scoperti=scoperti, espansi=r.espansi,
                                  generati=r.generati, secondi=secondi, completo=r.completo))
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
            ruoli_media=statistics.mean(r["ruoli"] for r in rs),
            scoperti_media=statistics.mean(r["scoperti"] for r in rs if r["scoperti"] is not None),
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

    kb = KnowledgeBase()
    righe = confronto(config.SEME, config.N_INCIDENTI_SCENARI, config.N_SCENARI_PER_DIMENSIONE,
                      lambda caso: adatti_con_kb(kb, caso))
    _scrivi_csv(righe, config.RESULTS_DIR / "search_confronto.csv")
    tabella = riepilogo(righe)
    _scrivi_csv(tabella, config.RESULTS_DIR / "search_riepilogo.csv")
    print(f"{'n':>3} {'ruoli':>6} {'algoritmo':12s} {'completi':>9s} {'espansi (media ± dev)':>24s} "
          f"{'secondi (media)':>16s}")
    for t in tabella:
        print(f"{t['dimensione']:>3} {t['ruoli_media']:>6.1f} {t['algoritmo']:12s} "
              f"{t['completi']:>4}/{t['casi']:<4} {t['espansi_media']:>12.1f} ± {t['espansi_dev']:<9.1f} "
              f"{t['secondi_media']:>16.6f}")
