"""Ricerca nello spazio degli stati: il percorso di un mezzo sulla mappa, dalla stazione all'incidente.

Confronto tra ricerca non informata e informata:
- BFS: frontiera FIFO, minimizza il numero di passi e non il tempo;
- LcFS: frontiera ordinata per costo, ottimale, nessuna euristica;
- A* con Manhattan: euristica dal problema semplificato senza ostacoli e senza celle lente;
- A* con h_PD: DB di pattern costruito con la programmazione dinamica sul problema semplificato
  senza le chiusure temporanee (cost_to_goal calcolata con LcFS sul grafo inverso dall'incidente).

Mappa: ostacoli permanenti (edifici), celle lente (centro storico, traffico) e chiusure temporanee
(strade chiuse il giorno della chiamata). Il costo di un passo è il costo della cella in cui si entra.
"""

import csv
import heapq
import itertools
import math
import random
import statistics
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config

INF = math.inf
Cella = tuple[int, int]
MOSSE = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass(frozen=True)
class Risultato:
    costo: float
    cammino: tuple | None
    espansi: int
    completo: bool  # False se la ricerca ha superato config.LIMITE_NODI


# --- Algoritmi ----

def _cammino(padre: dict, s) -> tuple:
    cammino = []
    while s is not None:
        cammino.append(s)
        s = padre[s]
    return tuple(reversed(cammino))


def a_star(inizio, obiettivo: Callable, successori: Callable, h: Callable,
           limite: int = config.LIMITE_NODI) -> Risultato:
    """A* con potatura dei cammini multipli (lista dei chiusi). Con h = 0 è LcFS.

    Il test di obiettivo è fatto alla selezione dalla frontiera. Con h consistente un nodo chiuso
    non viene mai riaperto, quindi il primo obiettivo estratto è ottimo (Proposizione 2.2).
    """
    conta = itertools.count()  # rompe i pareggi senza confrontare gli stati
    frontiera = [(h(inizio), 0, next(conta), inizio)]
    g_migliore = {inizio: 0}
    padre = {inizio: None}
    chiusi = set()
    espansi = 0
    while frontiera:
        _, g, _, s = heapq.heappop(frontiera)
        if s in chiusi:
            continue
        chiusi.add(s)
        if obiettivo(s):
            return Risultato(g, _cammino(padre, s), espansi, True)
        if espansi >= limite:
            return Risultato(INF, None, espansi, False)
        espansi += 1
        for s2, costo in successori(s):
            g2 = g + costo
            if g2 >= g_migliore.get(s2, INF):
                continue
            h2 = h(s2)
            if h2 == INF:
                continue
            g_migliore[s2] = g2
            padre[s2] = s
            heapq.heappush(frontiera, (g2 + h2, g2, next(conta), s2))
    return Risultato(INF, None, espansi, True)


def bfs(inizio, obiettivo: Callable, successori: Callable, limite: int = config.LIMITE_NODI) -> Risultato:
    """Ricerca in ampiezza con potatura dei cammini multipli. Trova il cammino con meno archi;
    il costo restituito è la somma dei costi di quel cammino, che può non essere il minimo."""
    frontiera = deque([(inizio, 0)])
    padre = {inizio: None}
    espansi = 0
    while frontiera:
        s, g = frontiera.popleft()
        if obiettivo(s):
            return Risultato(g, _cammino(padre, s), espansi, True)
        if espansi >= limite:
            return Risultato(INF, None, espansi, False)
        espansi += 1
        for s2, costo in successori(s):
            if s2 not in padre:
                padre[s2] = s
                frontiera.append((s2, g + costo))
    return Risultato(INF, None, espansi, True)


# --- Mappa ---
@dataclass(frozen=True)
class Mappa:
    righe: int
    colonne: int
    ostacoli: frozenset    # permanenti
    lente: frozenset       # celle che costano config.COSTO_CELLA_LENTA
    chiusure: frozenset    # temporanee: note solo al momento della chiamata

    def libera(self, c: Cella, con_chiusure: bool = True) -> bool:
        r, k = c
        dentro = 0 <= r < self.righe and 0 <= k < self.colonne
        return dentro and c not in self.ostacoli and not (con_chiusure and c in self.chiusure)

    def costo(self, c: Cella) -> int:
        return config.COSTO_CELLA_LENTA if c in self.lente else 1

    def vicini(self, c: Cella, con_chiusure: bool = True):
        r, k = c
        for dr, dk in MOSSE:
            v = (r + dr, k + dk)
            if self.libera(v, con_chiusure):
                yield v

    def successori(self, c: Cella):
        """Archi della mappa reale, con le chiusure. Il costo è quello della cella di arrivo."""
        for v in self.vicini(c):
            yield v, self.costo(v)

    def celle_libere(self) -> list:
        return [(r, k) for r in range(self.righe) for k in range(self.colonne) if self.libera((r, k))]


def genera_mappa(rng: random.Random, p_chiusure: float = config.PROB_CHIUSURA) -> Mappa:
    righe, colonne = config.DIM_GRIGLIA
    celle = [(r, k) for r in range(righe) for k in range(colonne)]
    ostacoli = frozenset(c for c in celle if rng.random() < config.PROB_OSTACOLO)
    libere = [c for c in celle if c not in ostacoli]
    lente = frozenset(c for c in libere if rng.random() < config.PROB_CELLA_LENTA)
    chiusure = frozenset(c for c in libere if rng.random() < p_chiusure)
    return Mappa(righe, colonne, ostacoli, lente, chiusure)


# --- Euristiche ---

def h_manhattan(obiettivo: Cella) -> Callable:
    """Problema semplificato: nessun ostacolo, nessuna chiusura, ogni cella costa 1 (il costo minimo).
    La sua soluzione ottima è in forma chiusa."""
    return lambda c: abs(c[0] - obiettivo[0]) + abs(c[1] - obiettivo[1])


def cost_to_goal(mappa: Mappa, obiettivo: Cella, con_chiusure: bool = False) -> tuple[dict, int]:
    """Programmazione dinamica: costo minimo da ogni cella all'obiettivo (docs/ricerca.md §3).

    LcFS sul grafo inverso, a partire dall'obiettivo. Sul grafo inverso l'arco m → n corrisponde
    all'arco n → m della mappa, che costa quanto la cella m. Restituisce la tabella e i nodi espansi.
    """
    tabella = {obiettivo: 0}
    frontiera = [(0, obiettivo)]
    espansi = 0
    while frontiera:
        d, m = heapq.heappop(frontiera)
        if d > tabella[m]:
            continue
        espansi += 1
        for n in mappa.vicini(m, con_chiusure):
            d2 = d + mappa.costo(m)
            if d2 < tabella.get(n, INF):
                tabella[n] = d2
                heapq.heappush(frontiera, (d2, n))
    return tabella, espansi


def h_pd(tabella: dict) -> Callable:
    """DB di pattern: il costo esatto del problema semplificato, letto dalla tabella.
    Una cella irraggiungibile anche senza chiusure lo è anche nella mappa reale: h = infinito."""
    return lambda c: tabella.get(c, INF)


# --- Interfaccia per il dispatch ----

def percorso(mappa: Mappa, partenza: Cella, arrivo: Cella, tabella: dict | None = None) -> Risultato:
    """Percorso ottimo sulla mappa reale con A* e h_PD. La tabella si calcola una volta per incidente."""
    if tabella is None:
        tabella, _ = cost_to_goal(mappa, arrivo)
    return a_star(partenza, lambda c: c == arrivo, mappa.successori, h_pd(tabella))


# --- Esperimento ---

ALGORITMI = ["BFS", "LcFS", "A* Manhattan", "A* PD"]


def confronto(seme: int = config.SEME) -> list[dict]:
    """Per ogni livello di chiusure, N_MAPPE mappe con un incidente e N_MEZZI_MAPPA mezzi.

    La tabella della PD è calcolata una volta per mappa (per incidente) e riusata per tutti i mezzi.
    """
    rng = random.Random(seme)
    righe = []
    for p in config.PROB_CHIUSURE:
        for k in range(config.N_MAPPE):
            mappa = genera_mappa(rng, p)
            libere = mappa.celle_libere()
            incidente = rng.choice(libere)
            t0 = time.perf_counter()
            tabella, espansi_pd = cost_to_goal(mappa, incidente)
            secondi_pd = time.perf_counter() - t0
            obiettivo = lambda c: c == incidente
            euristiche = {"LcFS": lambda c: 0, "A* Manhattan": h_manhattan(incidente), "A* PD": h_pd(tabella)}
            for m in range(config.N_MEZZI_MAPPA):
                stazione = rng.choice(libere)
                esiti = {"BFS": bfs(stazione, obiettivo, mappa.successori)}
                for nome, h in euristiche.items():
                    esiti[nome] = a_star(stazione, obiettivo, mappa.successori, h)
                ottimo = esiti["LcFS"].costo
                if any(esiti[a].costo != ottimo for a in ["A* Manhattan", "A* PD"]):
                    raise AssertionError("A* e LcFS con costi diversi: euristica non ammissibile")
                for nome, r in esiti.items():
                    righe.append(dict(chiusure=p, mappa=k, mezzo=m, algoritmo=nome, raggiungibile=ottimo < INF,
                                      costo=r.costo, ottimo=ottimo, espansi=r.espansi,
                                      espansi_pd=espansi_pd, secondi_pd=secondi_pd))
    return righe


def riepilogo(righe: list[dict]) -> list[dict]:
    """Media ± dev. std. per livello di chiusure e algoritmo, sui soli percorsi raggiungibili."""
    gruppi: dict = {}
    for r in righe:
        if r["raggiungibile"]:
            gruppi.setdefault((r["chiusure"], r["algoritmo"]), []).append(r)
    out = []
    for (p, alg), rs in sorted(gruppi.items()):
        espansi = [r["espansi"] for r in rs]
        eccesso = [100 * (r["costo"] - r["ottimo"]) / r["ottimo"] for r in rs if r["ottimo"] > 0]
        pd = {(r["mappa"]): r["espansi_pd"] for r in rs}
        out.append(dict(
            chiusure=p, algoritmo=alg, percorsi=len(rs),
            espansi_media=statistics.mean(espansi), espansi_dev=statistics.stdev(espansi),
            costo_in_piu_pct_media=statistics.mean(eccesso), costo_in_piu_pct_dev=statistics.stdev(eccesso),
            ottimi_pct=100 * sum(r["costo"] == r["ottimo"] for r in rs) / len(rs),
            espansi_pd_per_incidente=statistics.mean(pd.values()),
        ))
    return out


def _scrivi_csv(righe: list[dict], percorso_file: Path) -> None:
    percorso_file.parent.mkdir(parents=True, exist_ok=True)
    with percorso_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(righe[0].keys()))
        writer.writeheader()
        writer.writerows(righe)


if __name__ == "__main__":
    righe = confronto()
    _scrivi_csv(righe, config.RESULTS_DIR / "ricerca_percorsi.csv")
    tabella = riepilogo(righe)
    _scrivi_csv(tabella, config.RESULTS_DIR / "ricerca_riepilogo.csv")
    print(f"{'chiusure':>8} {'algoritmo':13s} {'espansi (media ± dev)':>24s} {'costo in più %':>15s} {'ottimi %':>9s}")
    for t in tabella:
        print(f"{t['chiusure']:>8.2f} {t['algoritmo']:13s} {t['espansi_media']:>12.1f} ± {t['espansi_dev']:<9.1f} "
              f"{t['costo_in_piu_pct_media']:>15.2f} {t['ottimi_pct']:>9.1f}")
    pd_media = statistics.mean(t["espansi_pd_per_incidente"] for t in tabella)
    print(f"Costo della PD: {pd_media:.0f} nodi espansi per incidente, riusati da {config.N_MEZZI_MAPPA} mezzi")
