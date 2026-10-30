"""Valutazione end-to-end: quanto pesa l'errore sulla gravità nel dispatch.

Su ogni scenario la squadra richiesta (KB Prolog) e l'assegnazione dei mezzi (simulated annealing,
src/assegnazioni.py) sono calcolate con la gravità stimata da una fonte, poi il piano viene giudicato
con la gravità vera. Il costo in più (regret) è misurato rispetto all'ottimo esatto con la gravità vera.

Fonti della gravità:
- vera:       regola di gravità con la velocità vera del fuoco (riferimento, non disponibile a runtime);
- ml:         regressione logistica dell'esperimento C (testo + registro), C scelto in CV;
- rete:       rete bayesiana come classificatore (testo + vulnerabilità);
- rete_kb:    la rete stima la velocità del fuoco, la KB applica la regola di gravità;
- prudente:   la KB assume la velocità alta (caso peggiore compatibile con le osservazioni).

Ipotesi: la KB riceve i campi veri della chiamata (l'operatore verifica la scheda), mentre ML e rete
vedono i campi estratti dal testo, come in addestramento. Le fonti differiscono quindi solo nella gravità.
"""

import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # noqa: E402

import bayes_engine as be  # noqa: E402
import config  # noqa: E402
import generate_dataset as gd  # noqa: E402
import assegnazioni as asg  # noqa: E402
import train_models as tm  # noqa: E402

FONTI = ["vera", "ml", "rete", "rete_kb", "prudente"]


# --- Stimatori addestrati sul dataset -----------------------------------------------------------

def addestra_stimatori(df: pd.DataFrame, seme: int = config.SEME) -> dict:
    """Regressione logistica dell'esperimento C (C scelto in CV) e rete bayesiana sull'intero dataset."""
    X = tm.campi_estratti(df["testo"])
    for col in tm.CAMPI_REGISTRO:
        X[col] = df[col].astype(str).values
    stimatore, griglia = tm.modelli(seme)["regressione_logistica"]
    ricerca = GridSearchCV(tm._pipeline(stimatore, list(X.columns)), griglia, scoring="f1_macro",
                           cv=StratifiedKFold(config.N_FOLD_CV, shuffle=True, random_state=seme))
    ricerca.fit(X, df["gravita"].values)
    return {"ml": ricerca.best_estimator_, "rete": be.stima(df), "colonne_ml": list(X.columns)}


def _evidenza(caso: asg.Caso) -> pd.DataFrame:
    """Campi estratti dal testo di ogni chiamata, più il registro della zona."""
    ids = sorted(caso.incidenti)
    righe = [caso.incidenti[i][0] for i in ids]
    ev = tm.campi_estratti(pd.Series([r["testo"] for r in righe]))
    ev["vulnerabilita"] = [r["vulnerabilita"] for r in righe]
    ev["adiacenti"] = [str(r["adiacenti"]) for r in righe]
    ev.index = ids
    return ev


def stime_gravita(stimatori: dict, caso: asg.Caso, kb) -> dict:
    """Per ogni fonte: {incidente: gravità} oppure le velocità da dare alla regola della KB."""
    ev = _evidenza(caso)
    ids = list(ev.index)
    ml = stimatori["ml"].predict(ev[stimatori["colonne_ml"]])
    rete = be.gravita_piu_probabile(stimatori["rete"], ev, be.OSSERVATE_CON_REGISTRO)
    vel = be.velocita_piu_probabile(stimatori["rete"], ev, be.OSSERVATE)
    return {
        "vera": {"gravita": None, "velocita": None},
        "ml": {"gravita": dict(zip(ids, ml)), "velocita": None},
        "rete": {"gravita": dict(zip(ids, rete)), "velocita": None},
        "rete_kb": {"gravita": None, "velocita": dict(zip(ids, vel))},
        "prudente": {"gravita": None, "velocita": {i: "alta" for i in ids}},
    }


def _con_velocita(caso: asg.Caso, velocita: dict | None) -> asg.Caso:
    """Copia del caso con la velocità del fuoco sostituita da una stima (per la regola della KB)."""
    if velocita is None:
        return caso
    incidenti = {i: ({**v, "velocita_fuoco": velocita[i]}, cella) for i, (v, cella) in caso.incidenti.items()}
    return asg.Caso(caso.mappa, caso.mezzi, incidenti)


# --- Giudizio di un piano con la gravità vera ----------------------------------------------------

def _abbinamento(mezzi: list, ruoli: list, adatti: dict) -> int:
    """Numero massimo di ruoli coperti dai mezzi, ognuno al più una volta (cammini aumentanti)."""
    assegnato = {}

    def prova(r, visti):
        for m in adatti.get(r, []):
            if m in mezzi and m not in visti:
                visti.add(m)
                if m not in assegnato or prova(assegnato[m], visti):
                    assegnato[m] = r
                    return True
        return False

    return sum(prova(r, set()) for r in ruoli)


def giudica(assegnazioni: list, adatti_veri: dict, costo: dict, penalita: float) -> dict:
    """Ruoli veri scoperti, mezzi inviati senza un ruolo vero, costo del piano con la gravità vera."""
    incidenti = {i for i, _ in adatti_veri} | {i for _, i, _ in assegnazioni}
    scoperti = superflui = 0
    for i in incidenti:
        mezzi = [m for m, j, _ in assegnazioni if j == i]
        ruoli = [r for r in adatti_veri if r[0] == i]
        coperti = _abbinamento(mezzi, ruoli, adatti_veri)
        scoperti += len(ruoli) - coperti
        superflui += len(mezzi) - coperti
    viaggio = sum(costo[(m, i)] for m, i, _ in assegnazioni)
    return dict(ruoli_veri_scoperti=scoperti, mezzi_superflui=superflui,
                costo_valutato=viaggio + penalita * scoperti)


# --- Esperimento ---------------------------------------------------------------------------------

def valuta(stimatori: dict, kb, n_incidenti: int = config.N_INCIDENTI_E2E,
           n_scenari: int = config.N_SCENARI_E2E, seme: int = config.SEME_E2E) -> list[dict]:
    rng = random.Random(seme)
    n_mezzi = math.ceil(n_incidenti * config.N_MEZZI_PER_INCIDENTE)
    righe = []
    for k in range(n_scenari):
        caso = asg.genera_caso(rng, n_incidenti, n_mezzi, variabili=lambda r: gd.genera_record(r, 0))
        costo = asg.tempi(caso)
        adatti_veri = asg.ruoli_dalla_kb(kb, caso)
        g_vera = {i: kb.gravita(i) for i in caso.incidenti}
        op_vera = {i: kb.gravita_operativa(i) for i in caso.incidenti}
        ottimo = asg.ottimo_esatto(asg.problema(adatti_veri, costo))
        for fonte, stima in stime_gravita(stimatori, caso, kb).items():
            caso_f = _con_velocita(caso, stima["velocita"])
            adatti = asg.ruoli_dalla_kb(kb, caso_f, stima["gravita"])
            g_stimata = {i: kb.gravita_operativa(i) for i in caso.incidenti}
            piano = asg.risolvi(asg.problema(adatti, costo), seme=k).assegnazioni
            esito = giudica(piano, adatti_veri, costo, config.PENALITA_RUOLO_SCOPERTO)
            righe.append(dict(
                scenario=k, fonte=fonte,
                gravita_corrette=sum(g_stimata[i] == op_vera[i] for i in caso.incidenti),
                critici_veri=sum(op_vera[i] == "critico" for i in caso.incidenti),
                critici_mancati=sum(op_vera[i] == "critico" and g_stimata[i] != "critico"
                                    for i in caso.incidenti),
                sovrastimati=sum(_livello(g_stimata[i]) > _livello(op_vera[i]) for i in caso.incidenti),
                ruoli_richiesti=len(adatti), ruoli_veri=len(adatti_veri),
                regret=esito["costo_valutato"] - ottimo, costo_ottimo=ottimo, **esito,
                incidenti=n_incidenti, gravita_base_vera=",".join(g_vera[i] for i in sorted(g_vera)),
            ))
    return righe


def _livello(g: str) -> int:
    return {"basso": 0, "alto": 1, "critico": 2}[g]


def riepilogo(righe: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(righe)
    df["accuratezza"] = df["gravita_corrette"] / df["incidenti"]
    metriche = ["accuratezza", "critici_mancati", "sovrastimati", "ruoli_veri_scoperti",
                "mezzi_superflui", "regret"]
    out = df.groupby("fonte")[metriche].agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out["critici_mancati_totale"] = df.groupby("fonte")["critici_mancati"].sum()
    out["critici_veri_totale"] = df.groupby("fonte")["critici_veri"].sum()
    return out.reindex(FONTI).reset_index()


if __name__ == "__main__":
    from prolog_engine import KnowledgeBase

    df = pd.read_csv(config.DATA_DIR / "dataset_sintetico.csv")
    stimatori = addestra_stimatori(df)
    righe = valuta(stimatori, KnowledgeBase())
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(righe).to_csv(config.RESULTS_DIR / "dispatch_e2e_scenari.csv", index=False)
    tab = riepilogo(righe)
    tab.to_csv(config.RESULTS_DIR / "dispatch_e2e_riepilogo.csv", index=False)
    pd.set_option("display.width", 200)
    print(tab.round(3).to_string(index=False))
