"""Rete bayesiana di SO-VVF-NEXT: 8 nodi, CPT per massima verosimiglianza con Laplace.

La struttura è il grafo causale di modello_generativo.py, senza il numero di edifici
adiacenti (non modellato, vedi docs/rete_bayesiana.md). Velocità del fuoco e vulnerabilità
sono nei dati di addestramento ma non sono osservate in inferenza: vengono marginalizzate.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pgmpy.estimators import BayesianEstimator  # noqa: E402
from pgmpy.inference import VariableElimination  # noqa: E402
from pgmpy.models import DiscreteBayesianNetwork  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning, module="pgmpy")

import config  # noqa: E402
from domain import (  # noqa: E402
    Gravita,
    Meteo,
    Persone,
    SostanzePericolose,
    TipoEdificio,
    VelocitaFuoco,
    Vulnerabilita,
)

STATI = {
    "tipo_edificio": [e.value for e in TipoEdificio],
    "piano": [str(p) for p in range(7)],
    "sostanze_pericolose": [e.value for e in SostanzePericolose],
    "meteo": [e.value for e in Meteo],
    "velocita_fuoco": [e.value for e in VelocitaFuoco],
    "persone": [e.value for e in Persone],
    "vulnerabilita": [e.value for e in Vulnerabilita],
    "gravita": [e.value for e in Gravita],
}

ARCHI = [
    ("tipo_edificio", "piano"),
    ("tipo_edificio", "sostanze_pericolose"),
    ("tipo_edificio", "persone"),
    ("piano", "persone"),
    ("sostanze_pericolose", "velocita_fuoco"),
    ("meteo", "velocita_fuoco"),
    ("persone", "gravita"),
    ("velocita_fuoco", "gravita"),
    ("sostanze_pericolose", "gravita"),
    ("vulnerabilita", "gravita"),
]

# Variabili osservabili dal testo della chiamata: sono l'evidenza dell'inferenza.
OSSERVATE = ["tipo_edificio", "piano", "sostanze_pericolose", "meteo", "persone"]


def prepara(df: pd.DataFrame) -> pd.DataFrame:
    """Converte le colonne in stringhe, come richiesto da pgmpy per gli stati discreti."""
    return pd.DataFrame({nodo: df[nodo].astype(str) for nodo in STATI})


def stima(df: pd.DataFrame, alpha: float = config.LAPLACE_ALPHA) -> DiscreteBayesianNetwork:
    """Stima le CPT per massima verosimiglianza con pseudoconteggi di Laplace (Dirichlet)."""
    dati = prepara(df)
    modello = DiscreteBayesianNetwork(ARCHI)
    stimatore = BayesianEstimator(modello, dati, state_names=STATI)
    modello.add_cpds(*[
        stimatore.estimate_cpd(nodo, prior_type="dirichlet", pseudo_counts=alpha)
        for nodo in STATI
    ])
    if not modello.check_model():
        raise ValueError("CPT non valide")
    return modello


def p_critico_righe(modello: DiscreteBayesianNetwork, df: pd.DataFrame) -> np.ndarray:
    """P(gravita = critico | osservate) per ogni riga. Usa solo le colonne OSSERVATE."""
    inferenza = VariableElimination(modello)
    dati = df[OSSERVATE].astype(str)
    indice = STATI["gravita"].index("critico")
    out = np.empty(len(dati))
    for k, riga in enumerate(dati[OSSERVATE].to_dict("records")):
        q = inferenza.query(["gravita"], evidence=riga, show_progress=False)
        out[k] = q.values[indice]
    return out


def p_critico_fuori_fold(df_stima: pd.DataFrame, df_evidenza: pd.DataFrame,
                         k: int = config.N_FOLD_CV, seme: int = config.SEME) -> np.ndarray:
    """P(critico) per ogni riga di addestramento, senza usare la propria etichetta.

    Ogni parte è prevista da una rete stimata sulle altre k-1 parti. df_stima contiene i valori
    veri (per stimare le CPT); df_evidenza contiene le osservazioni che il sistema vede a runtime
    (per l'inferenza). Serve come feature per il ML sui dati di addestramento.
    """
    out = np.empty(len(df_stima))
    y = df_stima["gravita"].values
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seme).split(df_stima, y):
        modello = stima(df_stima.iloc[tr])
        out[te] = p_critico_righe(modello, df_evidenza.iloc[te])
    return out


def probabilita(modello: DiscreteBayesianNetwork, nodo: str, stato, **genitori) -> float:
    """P(nodo = stato | genitori) letta dalla CPT appresa."""
    cpd = modello.get_cpds(nodo)
    valori = {nodo: str(stato), **{k: str(v) for k, v in genitori.items()}}
    indice = tuple(cpd.state_names[v].index(valori[v]) for v in cpd.variables)
    return float(cpd.values[indice])
