"""Assegnazione dei mezzi ai ruoli della squadra: ottimizzazione vincolata risolta con simulated annealing.

- Variabili: i ruoli richiesti dagli incidenti, derivati dalla KB Prolog (estinzione, rinforzo,
  soccorso in quota, decontaminazione).
- Domini: i mezzi adatti al ruolo secondo la KB e raggiungibili, più SCOPERTO (si chiede un rinforzo
  a un comando vicino, al costo config.PENALITA_RUOLO_SCOPERTO).
- Vincolo rigido: un mezzo copre al più un ruolo.
- Funzione obiettivo: tempo totale di percorrenza (da src/ricerca.py) più le penalità dei ruoli scoperti.

Come nel corso (§3.10), il vincolo rigido diventa un costo alto ma finito per ogni violazione:
la ricerca locale può attraversare assegnazioni non ammissibili, ma si restituisce sempre la migliore
assegnazione ammissibile visitata.
"""

import math
import random
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config
import modello_generativo as mg
import ricerca
from domain import TipoMezzo

INF = math.inf
SCOPERTO = None


# --- Scenario ------------------------------------------------------------------------------------

@dataclass
class Caso:
    mappa: ricerca.Mappa
    mezzi: dict       # id -> (TipoMezzo, cella)
    incidenti: dict   # id -> (variabili di modello_generativo, cella)


@dataclass
class Problema:
    ruoli: list       # (incidente, ruolo)
    domini: dict      # (incidente, ruolo) -> mezzi adatti e raggiungibili
    tempo: dict       # (mezzo, incidente) -> tempo di percorrenza
    penalita: float = config.PENALITA_RUOLO_SCOPERTO


@dataclass
class Soluzione:
    costo: float
    assegnazioni: list  # (mezzo, incidente, ruolo)
    scoperti: list      # (incidente, ruolo)


def genera_caso(rng: random.Random, n_incidenti: int, n_mezzi: int, variabili=None) -> Caso:
    """Mappa, flotta con la composizione di config e incidenti dal modello generativo."""
    mappa = ricerca.genera_mappa(rng)
    libere = mappa.celle_libere()
    tipi = [TipoMezzo(t) for t in config.COMPOSIZIONE_FLOTTA]
    pesi = list(config.COMPOSIZIONE_FLOTTA.values())
    mezzi = {f"m{k}": (rng.choices(tipi, pesi)[0], rng.choice(libere)) for k in range(n_mezzi)}
    genera = variabili or mg.genera_variabili
    incidenti = {f"i{k}": (genera(rng), rng.choice(libere)) for k in range(n_incidenti)}
    return Caso(mappa, mezzi, incidenti)


def tempi(caso: Caso) -> dict:
    """Tempo di ogni mezzo verso ogni incidente: A* con h_PD, una tabella PD per incidente."""
    out = {}
    for i, (_, arrivo) in caso.incidenti.items():
        tabella, _ = ricerca.cost_to_goal(caso.mappa, arrivo)
        for m, (_, partenza) in caso.mezzi.items():
            out[(m, i)] = ricerca.percorso(caso.mappa, partenza, arrivo, tabella).costo
    return out


def adiacenze(caso: Caso) -> list[tuple]:
    """Coppie di incidenti abbastanza vicini da propagare il pericolo."""
    ids = sorted(caso.incidenti)
    cella = {i: caso.incidenti[i][1] for i in ids}
    return [(a, b) for x, a in enumerate(ids) for b in ids[x + 1:]
            if ricerca.h_manhattan(cella[a])(cella[b]) <= config.DISTANZA_ADIACENZA]


def ruoli_dalla_kb(kb, caso: Caso, gravita_stimata: dict | None = None) -> dict:
    """Ruoli richiesti e mezzi adatti dalla KB Prolog. Senza gravita_stimata vale la regola con la velocità vera."""
    v = {i: var for i, (var, _) in caso.incidenti.items()}
    kb.carica_scenario(
        [dict(id=i, tipo=x["tipo_edificio"], piano=x["piano"], sostanze=x["sostanze_pericolose"],
              persone=x["persone"], meteo=x["meteo"]) for i, x in v.items()],
        {m: t for m, (t, _) in caso.mezzi.items()},
        {i: (x["vulnerabilita"], x["adiacenti"]) for i, x in v.items()},
        adiacenze(caso),
        {i: x["velocita_fuoco"] for i, x in v.items()},
        gravita_stimata,
    )
    return {(i, r): kb.mezzi_adatti(i, r) for i in sorted(caso.incidenti) for r in kb.squadra(i)}


def problema(adatti: dict, tempo: dict, penalita: float = config.PENALITA_RUOLO_SCOPERTO) -> Problema:
    """Domini = mezzi adatti secondo la KB e raggiungibili."""
    domini = {r: [m for m in mezzi if tempo.get((m, r[0]), INF) < INF] for r, mezzi in adatti.items()}
    return Problema(sorted(adatti), domini, tempo, penalita)


# --- Funzione obiettivo --------------------------------------------------------------------------

def conflitti(a: dict) -> int:
    """Violazioni del vincolo rigido: quante volte un mezzo è usato oltre la prima."""
    uso = Counter(m for m in a.values() if m is not SCOPERTO)
    return sum(n - 1 for n in uso.values())


def costo(p: Problema, a: dict) -> float:
    """Tempi + penalità dei ruoli scoperti + config.COSTO_CONFLITTO per ogni violazione del vincolo rigido."""
    totale = sum(p.penalita if m is SCOPERTO else p.tempo[(m, r[0])] for r, m in a.items())
    return totale + config.COSTO_CONFLITTO * conflitti(a)


def _soluzione(p: Problema, a: dict) -> Soluzione:
    return Soluzione(
        costo=costo(p, a),
        assegnazioni=[(m, i, r) for (i, r), m in a.items() if m is not SCOPERTO],
        scoperti=[(i, r) for (i, r), m in a.items() if m is SCOPERTO],
    )


# --- Simulated annealing con ripartenze casuali --------------------------------------------------

def simulated_annealing(p: Problema, rng: random.Random, passi: int = config.SA_PASSI,
                        t_iniziale: float = config.SA_T_INIZIALE, t_finale: float = config.SA_T_FINALE) -> Soluzione:
    """Un tentativo: assegnazione iniziale casuale, poi passi locali con raffreddamento geometrico.

    A ogni passo si sceglie a caso un ruolo e un nuovo valore del suo dominio. Il passo si accetta se
    non peggiora il costo, altrimenti con probabilità exp(-(h' - h) / T). Si restituisce la migliore
    assegnazione ammissibile visitata (al peggio, tutti i ruoli scoperti).
    """
    valori = {r: p.domini[r] + [SCOPERTO] for r in p.ruoli}
    migliore = {r: SCOPERTO for r in p.ruoli}
    h_migliore = costo(p, migliore)
    a = {r: rng.choice(valori[r]) for r in p.ruoli}
    uso = Counter(m for m in a.values() if m is not SCOPERTO)  # quante volte è usato ogni mezzo
    h = costo(p, a)

    def valore(r, m):
        return p.penalita if m is SCOPERTO else p.tempo[(m, r[0])]

    raffreddamento = (t_finale / t_iniziale) ** (1 / passi)
    t = t_iniziale
    for _ in range(passi):
        r = rng.choice(p.ruoli)
        vecchio, nuovo = a[r], rng.choice(valori[r])
        if nuovo == vecchio:
            t *= raffreddamento
            continue
        # Variazione del costo: cambia solo il ruolo r, quindi basta guardare i due mezzi coinvolti.
        delta = valore(r, nuovo) - valore(r, vecchio)
        if vecchio is not SCOPERTO and uso[vecchio] > 1:
            delta -= config.COSTO_CONFLITTO  # si libera un mezzo conteso
        if nuovo is not SCOPERTO and uso[nuovo] > 0:
            delta += config.COSTO_CONFLITTO  # si usa un mezzo già occupato
        if delta <= 0 or rng.random() < math.exp(-delta / t):
            a[r] = nuovo
            if vecchio is not SCOPERTO:
                uso[vecchio] -= 1
            if nuovo is not SCOPERTO:
                uso[nuovo] += 1
            h += delta
            if h < h_migliore and all(n <= 1 for n in uso.values()):
                migliore, h_migliore = dict(a), h
        t *= raffreddamento
    return _soluzione(p, migliore)


def risolvi(p: Problema, seme: int = config.SEME, ripartenze: int = config.SA_RIPARTENZE) -> Soluzione:
    """Ripartenza casuale: più tentativi indipendenti, si tiene il migliore."""
    rng = random.Random(seme)
    return min((simulated_annealing(p, rng) for _ in range(ripartenze)), key=lambda s: s.costo)


def ottimo_esatto(p: Problema) -> float:
    """Costo minimo esatto con l'algoritmo ungherese (assegnamento ruoli-mezzi, una colonna "scoperto"
    per ruolo). Serve solo da riferimento per valutare il simulated annealing."""
    if not p.ruoli:
        return 0.0
    mezzi = sorted({m for d in p.domini.values() for m in d})
    c = np.full((len(p.ruoli), len(mezzi) + len(p.ruoli)), 1e9)
    for x, r in enumerate(p.ruoli):
        for y, m in enumerate(mezzi):
            if m in p.domini[r]:
                c[x, y] = p.tempo[(m, r[0])]
        c[x, len(mezzi) + x] = p.penalita
    righe, colonne = linear_sum_assignment(c)
    return float(c[righe, colonne].sum())


# --- Esperimento ---------------------------------------------------------------------------------

def valuta(kb, seme: int = config.SEME) -> list[dict]:
    """Distanza dall'ottimo del simulated annealing, con un tentativo e con le ripartenze."""
    rng = random.Random(seme)
    righe = []
    for n in config.N_INCIDENTI_SCENARI:
        n_mezzi = math.ceil(n * config.N_MEZZI_PER_INCIDENTE)
        for k in range(config.N_SCENARI_PER_DIMENSIONE):
            caso = genera_caso(rng, n, n_mezzi)
            p = problema(ruoli_dalla_kb(kb, caso), tempi(caso))
            ottimo = ottimo_esatto(p)
            for nome, ripartenze in [("1 tentativo", 1), (f"{config.SA_RIPARTENZE} tentativi", config.SA_RIPARTENZE)]:
                s = risolvi(p, seme=k, ripartenze=ripartenze)
                righe.append(dict(incidenti=n, scenario=k, ruoli=len(p.ruoli), variante=nome, costo=s.costo,
                                  ottimo=ottimo, scarto_pct=100 * (s.costo - ottimo) / ottimo if ottimo else 0.0))
    return righe


def riepilogo(righe: list[dict]) -> list[dict]:
    gruppi: dict = {}
    for r in righe:
        gruppi.setdefault((r["incidenti"], r["variante"]), []).append(r)
    return [dict(incidenti=n, variante=v, ruoli_media=statistics.mean(r["ruoli"] for r in rs),
                 ottimi_pct=100 * sum(r["costo"] == r["ottimo"] for r in rs) / len(rs),
                 scarto_pct_media=statistics.mean(r["scarto_pct"] for r in rs),
                 scarto_pct_dev=statistics.stdev(r["scarto_pct"] for r in rs))
            for (n, v), rs in sorted(gruppi.items())]


if __name__ == "__main__":
    import pandas as pd
    from prolog_engine import KnowledgeBase

    righe = valuta(KnowledgeBase())
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(righe).to_csv(config.RESULTS_DIR / "assegnazioni_scenari.csv", index=False)
    tabella = pd.DataFrame(riepilogo(righe))
    tabella.to_csv(config.RESULTS_DIR / "assegnazioni_riepilogo.csv", index=False)
    print(tabella.round(2).to_string(index=False))
