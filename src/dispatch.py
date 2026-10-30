"""Assegnazione dei mezzi ai ruoli della squadra come problema di soddisfacimento di vincoli (CSP).

- Variabili: i ruoli richiesti dagli incidenti (estinzione, rinforzo, soccorso in quota,
  decontaminazione), derivati dalla KB Prolog a partire dalla gravità operativa.
- Domini: i mezzi adatti al ruolo secondo la KB e raggiungibili, più il valore "scoperto"
  (si chiede un rinforzo a un comando vicino, al costo config.PENALITA_RUOLO_SCOPERTO).
- Vincolo: un mezzo copre al più un ruolo.
- Obiettivo: minimizzare il tempo totale di percorrenza, dato da src/ricerca.py, più le penalità.

Risolutore: branch and bound in profondità. Ordine delle variabili MRV (prima il ruolo con meno mezzi
liberi), valori dal più economico, domini aggiornati a ogni assegnazione. Il limite inferiore viene da due
rilassamenti del vincolo di esclusività (mezzi contesi e capacità), vedi stima().
"""

import csv
import math
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config
import modello_generativo as mg
import ricerca
from domain import TipoMezzo

INF = math.inf


# --- Scenario ------------------------------------------------------------------------------------

@dataclass
class Caso:
    mappa: ricerca.Mappa
    mezzi: dict         # id -> (TipoMezzo, cella)
    incidenti: dict     # id -> (variabili di modello_generativo, cella)


@dataclass
class Scenario:
    ruoli: tuple        # (incidente, ruolo)
    mezzi: tuple
    adatti: dict        # (incidente, ruolo) -> mezzi adatti secondo la KB
    costo: dict         # (mezzo, incidente) -> tempo di percorrenza, INF se irraggiungibile
    penalita: float = config.PENALITA_RUOLO_SCOPERTO


@dataclass
class Soluzione:
    costo: float
    assegnazioni: list  # (mezzo, incidente, ruolo)
    scoperti: list      # (incidente, ruolo)
    nodi: int
    completa: bool      # False se la ricerca ha superato il limite di nodi


def genera_caso(rng: random.Random, n_incidenti: int, n_mezzi: int, variabili: Callable | None = None) -> Caso:
    """Mappa, flotta con la composizione di config e incidenti dal modello generativo."""
    mappa = ricerca.genera_mappa(rng)
    libere = mappa.celle_libere()
    tipi = [TipoMezzo(t) for t in config.COMPOSIZIONE_FLOTTA]
    pesi = list(config.COMPOSIZIONE_FLOTTA.values())
    mezzi = {f"m{k}": (rng.choices(tipi, pesi)[0], rng.choice(libere)) for k in range(n_mezzi)}
    genera = variabili or mg.genera_variabili
    incidenti = {f"i{k}": (genera(rng), rng.choice(libere)) for k in range(n_incidenti)}
    return Caso(mappa, mezzi, incidenti)


def costi_caso(caso: Caso) -> dict:
    """Tempo di percorrenza di ogni mezzo verso ogni incidente: A* con h_PD sulla mappa reale.
    La tabella della PD è calcolata una volta per incidente e riusata per tutti i mezzi."""
    costo = {}
    for i, (_, arrivo) in caso.incidenti.items():
        tabella, _ = ricerca.cost_to_goal(caso.mappa, arrivo)
        for m, (_, partenza) in caso.mezzi.items():
            costo[(m, i)] = ricerca.percorso(caso.mappa, partenza, arrivo, tabella).costo
    return costo


def adiacenze_caso(caso: Caso, soglia: int = config.DISTANZA_ADIACENZA) -> list[tuple]:
    """Coppie di incidenti abbastanza vicini da propagare il pericolo."""
    ids = sorted(caso.incidenti)
    cella = {i: caso.incidenti[i][1] for i in ids}
    return [(a, b) for x, a in enumerate(ids) for b in ids[x + 1:]
            if abs(cella[a][0] - cella[b][0]) + abs(cella[a][1] - cella[b][1]) <= soglia]


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


def scenario_da_adatti(adatti: dict, mezzi: tuple, costo: dict,
                       penalita: float = config.PENALITA_RUOLO_SCOPERTO) -> Scenario:
    return Scenario(tuple(sorted(adatti)), tuple(mezzi), adatti, costo, penalita)


# --- Branch and bound ----------------------------------------------------------------------------

class _LimiteSuperato(Exception):
    pass


def domini(sc: Scenario) -> dict:
    """Dominio di ogni ruolo: mezzi adatti secondo la KB e raggiungibili, dal più vicino."""
    return {r: sorted((m for m in sc.adatti.get(r, []) if sc.costo.get((m, r[0]), INF) < INF),
                      key=lambda m: sc.costo[(m, r[0])])
            for r in sc.ruoli}


def valori(sc: Scenario, dominio: dict, r, usati) -> list:
    """Valori ancora ammessi per il ruolo r, dal più economico: (costo, mezzo) e (penalità, None) = scoperto.
    Escludere i mezzi già usati aggiorna il dominio dopo ogni assegnazione (forward checking)."""
    v = [(sc.costo[(m, r[0])], m) for m in dominio[r] if m not in usati] + [(sc.penalita, None)]
    return sorted(v, key=lambda x: (x[0], x[1] is None, x[1] or ""))


def stima(sc: Scenario, dominio: dict, residui, usati) -> float:
    """Limite inferiore del costo dei ruoli residui (docs/ricerca.md §4).

    Base (h_libero): ogni ruolo prende il suo valore più economico, anche se un altro ruolo usa
    lo stesso mezzo. Alla base si sommano due correzioni ammissibili, su gruppi di ruoli disgiunti:
    - capacità: se k ruoli possono usare solo un gruppo di d mezzi e k > d, almeno k - d restano
      scoperti; si scelgono come scoperti quelli per cui la penalità aggiunge meno;
    - mezzi contesi, sugli altri ruoli: se più ruoli hanno come prima scelta lo stesso mezzo,
      uno solo può averlo, gli altri pagano almeno il loro secondo valore.
    """
    base, primo, secondo, prima_scelta, liberi = 0, {}, {}, {}, {}
    for r in residui:
        v = valori(sc, dominio, r, usati)
        primo[r], prima_scelta[r] = v[0]
        secondo[r] = v[1][0] if len(v) > 1 else v[0][0]
        liberi[r] = frozenset(m for _, m in v if m is not None)
        base += primo[r]

    def contesi(ruoli) -> float:
        aumenti: dict = {}
        for r in ruoli:
            if prima_scelta[r] is not None:
                aumenti.setdefault(prima_scelta[r], []).append(secondo[r] - primo[r])
        return sum(sum(a) - max(a) for a in aumenti.values())

    migliore = contesi(residui)
    for mezzi in set(liberi.values()):
        dentro = [r for r in residui if liberi[r] <= mezzi]
        eccesso = len(dentro) - len(mezzi)
        if eccesso > 0:
            capacita = sum(sorted(sc.penalita - primo[r] for r in dentro)[:eccesso])
            migliore = max(migliore, capacita + contesi([r for r in residui if r not in dentro]))
    return base + migliore


def risolvi(sc: Scenario, usa_stima: bool = True, limite_nodi: int = config.LIMITE_NODI) -> Soluzione:
    """Branch and bound in profondità. Senza stima (usa_stima=False) pota solo con il costo già speso:
    è il riferimento non informato, come LcFS lo è per A*."""
    dominio = domini(sc)
    migliore = {"costo": INF, "piano": None}
    nodi = 0

    def priorita(r, usati):
        """MRV; a parità di valori, prima il ruolo con il rimpianto più alto (secondo - primo valore)."""
        v = valori(sc, dominio, r, usati)
        rimpianto = (v[1][0] - v[0][0]) if len(v) > 1 else 0
        return len(v), -rimpianto, r

    def esplora(residui: tuple, usati: frozenset, costo: float, piano: list):
        nonlocal nodi
        nodi += 1
        if nodi > limite_nodi:
            raise _LimiteSuperato
        if not residui:
            if costo < migliore["costo"]:
                migliore["costo"], migliore["piano"] = costo, list(piano)
            return
        if costo + (stima(sc, dominio, residui, usati) if usa_stima else 0) >= migliore["costo"]:
            return  # nessun completamento può battere la soluzione migliore trovata finora
        r = min(residui, key=lambda x: priorita(x, usati))
        resto = tuple(x for x in residui if x != r)
        for c, m in valori(sc, dominio, r, usati):
            esplora(resto, usati | {m} if m else usati, costo + c, piano + [(m, r[0], r[1])])

    try:
        esplora(tuple(sc.ruoli), frozenset(), 0, [])
        completa = True
    except _LimiteSuperato:
        completa = False
    piano = migliore["piano"] or []
    return Soluzione(
        costo=migliore["costo"] if completa else INF,
        assegnazioni=[(m, i, r) for m, i, r in piano if m is not None],
        scoperti=[(i, r) for m, i, r in piano if m is None],
        nodi=nodi, completa=completa,
    )


# --- Esperimento ---------------------------------------------------------------------------------

def confronto(seme: int, adatti_fn: Callable) -> list[dict]:
    """Branch and bound con e senza stima, su N_SCENARI_PER_DIMENSIONE scenari per dimensione."""
    rng = random.Random(seme)
    righe = []
    for n in config.N_INCIDENTI_SCENARI:
        n_mezzi = math.ceil(n * config.N_MEZZI_PER_INCIDENTE)
        for k in range(config.N_SCENARI_PER_DIMENSIONE):
            caso = genera_caso(rng, n, n_mezzi)
            sc = scenario_da_adatti(adatti_fn(caso), tuple(caso.mezzi), costi_caso(caso))
            esiti = {"senza stima": risolvi(sc, usa_stima=False), "con stima": risolvi(sc, usa_stima=True)}
            costi = {s.costo for s in esiti.values() if s.completa}
            if len(costi) > 1:
                raise AssertionError(f"soluzioni complete con costi diversi: {costi}")
            for nome, s in esiti.items():
                righe.append(dict(incidenti=n, scenario=k, ruoli=len(sc.ruoli), mezzi=n_mezzi, algoritmo=nome,
                                  costo=s.costo, scoperti=len(s.scoperti), nodi=s.nodi, completa=s.completa))
    return righe


def riepilogo(righe: list[dict]) -> list[dict]:
    gruppi: dict = {}
    for r in righe:
        gruppi.setdefault((r["incidenti"], r["algoritmo"]), []).append(r)
    out = []
    for (n, alg), rs in sorted(gruppi.items()):
        nodi = [r["nodi"] for r in rs]
        out.append(dict(
            incidenti=n, algoritmo=alg, casi=len(rs), completi=sum(r["completa"] for r in rs),
            ruoli_media=statistics.mean(r["ruoli"] for r in rs),
            scoperti_media=statistics.mean(r["scoperti"] for r in rs),
            nodi_media=statistics.mean(nodi), nodi_dev=statistics.stdev(nodi),
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
    righe = confronto(config.SEME, lambda caso: adatti_con_kb(kb, caso))
    _scrivi_csv(righe, config.RESULTS_DIR / "dispatch_confronto.csv")
    tabella = riepilogo(righe)
    _scrivi_csv(tabella, config.RESULTS_DIR / "dispatch_riepilogo.csv")
    print(f"{'n':>3} {'ruoli':>6} {'algoritmo':13s} {'completi':>9s} {'nodi (media ± dev)':>26s}")
    for t in tabella:
        print(f"{t['incidenti']:>3} {t['ruoli_media']:>6.1f} {t['algoritmo']:13s} {t['completi']:>4}/{t['casi']:<4} "
              f"{t['nodi_media']:>13.1f} ± {t['nodi_dev']:<10.1f}")
