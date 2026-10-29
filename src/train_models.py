"""Classificazione della gravità con nested cross-validation (esperimenti A e B).

Esperimento A: i campi estratti dal testo dall'adattatore NLP.
Esperimento B: A, più i fatti del registro (vulnerabilità, edifici adiacenti) e la
probabilità di gravità critica della rete bayesiana (binned).

La regola di gravità non entra mai nelle feature (target leakage). La rete bayesiana
per i dati di addestramento è stimata fuori fold, così la feature non contiene l'etichetta.

Esecuzione dalla radice del progetto: python src/train_models.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # noqa: E402
from sklearn.naive_bayes import MultinomialNB  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402

import bayes_engine as be  # noqa: E402
import config  # noqa: E402
from nlp_extract import estrai_campi  # noqa: E402

CAMPI_TESTO = ["tipo_edificio", "piano", "sostanze_pericolose", "persone", "meteo"]
CAMPI_REGISTRO = ["vulnerabilita", "adiacenti"]
BIN_PROB = [0.1, 0.2, 0.4, 0.6]  # estremi interni delle classi di P(critico)


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
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), colonne)])
    return Pipeline([("pre", pre), ("clf", stimatore)])


def _fold_esterni(y: np.ndarray, seme: int):
    return StratifiedKFold(config.N_FOLD_CV, shuffle=True, random_state=seme).split(np.zeros(len(y)), y)


def _valuta(esperimento: str, X: pd.DataFrame, y: np.ndarray, colonne: list,
            probabilita: list | None, seme: int) -> list[dict]:
    """Nested CV: esterna per la stima, interna per gli iperparametri (scoring macro F1).

    probabilita: per ogni fold esterno, (p_tr, p_te) della rete bayesiana. Se presente,
    la probabilità binned diventa una feature categoriale 'p_bin'.
    """
    righe = []
    for fold, (tr, te) in enumerate(_fold_esterni(y, seme)):
        X_tr, X_te = X.iloc[tr].copy(), X.iloc[te].copy()
        y_tr, y_te = y[tr], y[te]
        if probabilita is not None:
            p_tr, p_te = probabilita[fold]
            X_tr["p_bin"] = np.digitize(p_tr, BIN_PROB).astype(str)
            X_te["p_bin"] = np.digitize(p_te, BIN_PROB).astype(str)
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
                accuracy=accuracy_score(y_te, pred),
                f1_macro=f1_score(y_te, pred, average="macro"),
                parametri=str(parametri),
            ))
    return righe


def esegui(df: pd.DataFrame, seme: int = config.SEME) -> list[dict]:
    """Esperimenti A e B su un dataset. Restituisce una riga per modello e fold esterno."""
    y = df["gravita"].values
    evidenza = campi_estratti(df["testo"])

    # Esperimento A: solo i campi del testo.
    righe = _valuta("A", evidenza, y, CAMPI_TESTO, None, seme)

    # Esperimento B: A, più registro, più P(critico) della rete. La rete per le righe di
    # addestramento è stimata fuori fold; per le righe di test, sul solo fold di addestramento.
    X_b = evidenza.copy()
    for col in CAMPI_REGISTRO:
        X_b[col] = df[col].astype(str).values
    probabilita = []
    for tr, te in _fold_esterni(y, seme):
        p_tr = be.p_critico_fuori_fold(df.iloc[tr], evidenza.iloc[tr], seme=seme)
        rete = be.stima(df.iloc[tr])
        p_te = be.p_critico_righe(rete, evidenza.iloc[te])
        probabilita.append((p_tr, p_te))
    righe.extend(_valuta("B", X_b, y, CAMPI_TESTO + CAMPI_REGISTRO, probabilita, seme))

    # Controlli, non realizzabili in pratica. C: come B senza la rete, isola il contributo della rete.
    # Oracolo: come C con la velocità vera del fuoco, misura quanto pesa la variabile nascosta.
    righe.extend(_valuta("C", X_b, y, CAMPI_TESTO + CAMPI_REGISTRO, None, seme))
    X_o = X_b.copy()
    X_o["velocita_fuoco"] = df["velocita_fuoco"].astype(str).values
    righe.extend(_valuta("oracolo", X_o, y, CAMPI_TESTO + CAMPI_REGISTRO + ["velocita_fuoco"], None, seme))
    return righe


def riepilogo(righe: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(righe)
    gruppi = df.groupby(["esperimento", "modello"])
    out = gruppi.agg(
        accuracy_media=("accuracy", "mean"), accuracy_dev=("accuracy", "std"),
        f1_media=("f1_macro", "mean"), f1_dev=("f1_macro", "std"),
    ).reset_index()
    return out


if __name__ == "__main__":
    df = pd.read_csv(config.DATA_DIR / "dataset_sintetico.csv")
    righe = esegui(df)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(righe).to_csv(config.RESULTS_DIR / "ml_esiti_fold.csv", index=False)
    tab = riepilogo(righe)
    tab.to_csv(config.RESULTS_DIR / "ml_riepilogo.csv", index=False)
    print(tab.round(3).to_string(index=False))
