"""Classificazione della gravità con nested cross-validation.

Esperimento A: i campi estratti dal testo dall'adattatore NLP.
Esperimento B: A, più i fatti del registro (vulnerabilità, edifici adiacenti) e la
probabilità di gravità critica della rete bayesiana, discretizzata in fasce a quantili.
Rete: la rete bayesiana usata come classificatore, con la stessa evidenza di B tranne gli adiacenti.
Controlli: C = B senza la rete; oracolo = C con la velocità vera del fuoco.

La regola di gravità non entra mai nelle feature (target leakage). La rete bayesiana
per i dati di addestramento è stimata fuori fold, così la feature non contiene l'etichetta.

Esecuzione dalla radice del progetto: python src/train_models.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

import bayes_engine as be
import config
from nlp_extract import estrai_campi

CAMPI_TESTO = ["tipo_edificio", "piano", "sostanze_pericolose", "persone", "meteo"]
CAMPI_REGISTRO = ["vulnerabilita", "adiacenti"]
N_FASCE_PROB = 5  # fasce a quantili di P(critico), calcolate sul fold di addestramento


def campi_estratti(testo: pd.Series) -> pd.DataFrame:
    """Campi ricavati dal testo. I campi non riconosciuti diventano la categoria 'nessuno'."""
    ev = pd.DataFrame([estrai_campi(t) for t in testo])[CAMPI_TESTO]
    ev = ev.fillna("nessuno")
    ev["piano"] = ev["piano"].astype(str)
    return ev


def modelli(seme: int) -> dict:
    """Modello, griglia degli iperparametri e nome. Il baseline non ha iperparametri."""
    return {
        "baseline_maggioranza": (DummyClassifier(strategy="most_frequent"), {}),
        "regressione_logistica": (LogisticRegression(max_iter=2000),
                                  {"clf__C": [0.1, 1, 10]}),
        "random_forest": (RandomForestClassifier(n_estimators=200, random_state=seme),
                          {"clf__max_depth": [3, 6, None]}),
        "naive_bayes": (MultinomialNB(), {"clf__alpha": [0.1, 1, 10]}),
    }


def _pipeline(stimatore, colonne: list) -> Pipeline:
    """One-hot sulle colonne categoriali; P(critico), se presente, discretizzata a quantili.

    Le fasce a quantili evitano soglie scelte a mano e danno a naive Bayes multinomiale
    feature non negative come agli altri modelli.
    """
    parti = [("cat", OneHotEncoder(handle_unknown="ignore"), [c for c in colonne if c != "p_critico"])]
    if "p_critico" in colonne:
        parti.append(("prob", KBinsDiscretizer(N_FASCE_PROB, encode="onehot", strategy="quantile",
                                               quantile_method="averaged_inverted_cdf"), ["p_critico"]))
    return Pipeline([("pre", ColumnTransformer(parti)), ("clf", stimatore)])


METRICHE = ["accuracy", "f1_macro", "f1_pesata"]


def metriche(y_vero, y_pred) -> dict:
    """Accuratezza, F1 macro (ogni classe pesa uguale) e F1 pesata (ogni classe pesa quanto la sua frequenza)."""
    return dict(
        accuracy=accuracy_score(y_vero, y_pred),
        f1_macro=f1_score(y_vero, y_pred, average="macro"),
        f1_pesata=f1_score(y_vero, y_pred, average="weighted"),
    )


def riassumi(righe: list[dict], chiavi: list) -> pd.DataFrame:
    """Media e deviazione standard di ogni metrica sui fold esterni."""
    gruppi = pd.DataFrame(righe).groupby(chiavi)
    aggregati = {}
    for m in METRICHE:
        aggregati[f"{m}_media"] = (m, "mean")
        aggregati[f"{m}_dev"] = (m, "std")
    return gruppi.agg(**aggregati).reset_index()


def _fold_esterni(y: np.ndarray, seme: int):
    return StratifiedKFold(config.N_FOLD_CV, shuffle=True, random_state=seme).split(np.zeros(len(y)), y)


def _valuta(esperimento: str, X: pd.DataFrame, y: np.ndarray, colonne: list,
            probabilita: list | None, seme: int) -> list[dict]:
    """Nested CV: esterna per la stima, interna per gli iperparametri (scoring macro F1).

    probabilita: per ogni fold esterno, (p_tr, p_te) della rete bayesiana. Se presente,
    diventa la feature 'p_critico', discretizzata dentro la pipeline.
    """
    righe = []
    for fold, (tr, te) in enumerate(_fold_esterni(y, seme)):
        X_tr, X_te = X.iloc[tr].copy(), X.iloc[te].copy()
        y_tr, y_te = y[tr], y[te]
        if probabilita is not None:
            p_tr, p_te = probabilita[fold]
            X_tr["p_critico"] = p_tr
            X_te["p_critico"] = p_te
        for nome, (stimatore, griglia) in modelli(seme).items():
            pipe = _pipeline(stimatore, list(X_tr.columns))
            if griglia:
                interna = StratifiedKFold(config.N_FOLD_CV, shuffle=True, random_state=seme)
                ricerca = GridSearchCV(pipe, griglia, cv=interna, scoring="f1_macro")
                ricerca.fit(X_tr, y_tr)
                modello, parametri = ricerca.best_estimator_, ricerca.best_params_
            else:
                modello = pipe.fit(X_tr, y_tr)
                parametri = {}
            pred = modello.predict(X_te)
            righe.append(dict(
                esperimento=esperimento, modello=nome, fold=fold,
                **metriche(y_te, pred),
                parametri=str(parametri),
            ))
    return righe


def esegui(df: pd.DataFrame, seme: int = config.SEME) -> list[dict]:
    """Esperimenti A e B su un dataset. Restituisce una riga per modello e fold esterno."""
    y = df["gravita"].values
    evidenza = campi_estratti(df["testo"])

    # Esperimento A: solo i campi del testo.
    righe = _valuta("A", evidenza, y, CAMPI_TESTO, None, seme)

    # Esperimento B: A, più registro, più P(critico) della rete, con la vulnerabilità come evidenza.
    # Per ogni fold esterno alpha di Laplace è scelto in CV interna; la rete per le righe di
    # addestramento è stimata fuori fold, per le righe di test sul solo fold di addestramento.
    X_b = evidenza.copy()
    for col in CAMPI_REGISTRO:
        X_b[col] = df[col].astype(str).values
    ev_rete = evidenza.copy()
    ev_rete["vulnerabilita"] = df["vulnerabilita"].astype(str).values
    colonne_rete = be.OSSERVATE_CON_REGISTRO
    probabilita, righe_rete = [], []
    for fold, (tr, te) in enumerate(_fold_esterni(y, seme)):
        alpha = be.scegli_alpha(df.iloc[tr], ev_rete.iloc[tr], colonne_rete, seme=seme)
        p_tr = be.p_critico_fuori_fold(df.iloc[tr], ev_rete.iloc[tr], seme=seme,
                                       colonne=colonne_rete, alpha=alpha)
        rete = be.stima(df.iloc[tr], alpha)
        probabilita.append((p_tr, be.p_critico_righe(rete, ev_rete.iloc[te], colonne_rete)))
        pred = be.gravita_piu_probabile(rete, ev_rete.iloc[te], colonne_rete)
        righe_rete.append(dict(
            esperimento="rete", modello="rete_bayesiana", fold=fold,
            **metriche(y[te], pred),
            parametri=str({"alpha": alpha}),
        ))
    righe.extend(_valuta("B", X_b, y, CAMPI_TESTO + CAMPI_REGISTRO, probabilita, seme))
    righe.extend(righe_rete)

    # Controlli, non realizzabili in pratica. C: come B senza la rete, isola il contributo della rete.
    # Oracolo: come C con la velocità vera del fuoco, misura quanto pesa la variabile nascosta.
    righe.extend(_valuta("C", X_b, y, CAMPI_TESTO + CAMPI_REGISTRO, None, seme))
    X_o = X_b.copy()
    X_o["velocita_fuoco"] = df["velocita_fuoco"].astype(str).values
    righe.extend(_valuta("oracolo", X_o, y, CAMPI_TESTO + CAMPI_REGISTRO + ["velocita_fuoco"], None, seme))
    return righe


def riepilogo(righe: list[dict]) -> pd.DataFrame:
    return riassumi(righe, ["esperimento", "modello"])


if __name__ == "__main__":
    df = pd.read_csv(config.DATA_DIR / "dataset_sintetico.csv")
    righe = esegui(df)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(righe).to_csv(config.RESULTS_DIR / "ml_esiti_fold.csv", index=False)
    tab = riepilogo(righe)
    tab.to_csv(config.RESULTS_DIR / "ml_riepilogo.csv", index=False)
    print(tab.round(3).to_string(index=False))
