"""Rete bayesiana con 8 nodi, CPT per massima verosimiglianza con Laplace.

La struttura è il grafo causale di modello_generativo.py, senza il numero di edifici
adiacenti (non modellato).  Velocità del fuoco e vulnerabilità
sono nei dati di addestramento ma non sono osservate in inferenza: vengono marginalizzate.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import warnings

# Filtro su un warning di pgmpy su alcune classi deprecate
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*pgmpy\..*")

import numpy as np
import pandas as pd
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork
from sklearn.model_selection import StratifiedKFold


import config
from domain import (
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
# Con il registro della zona è nota anche la vulnerabilità (esperimento B, dispatch).
OSSERVATE_CON_REGISTRO = OSSERVATE + ["vulnerabilita"]
ALPHA_CANDIDATI = [0.1, 1.0, 10.0]  # griglia di alpha di Laplace per la CV interna


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


def _query(modello: DiscreteBayesianNetwork, df: pd.DataFrame, colonne: list, nodo: str) -> np.ndarray:
    """P(nodo | colonne) per ogni riga, una colonna per stato di STATI[nodo].

    Le righe con la stessa evidenza condividono la query: le combinazioni distinte sono poche.
    """
    inferenza = VariableElimination(modello)
    dati = df[colonne].astype(str)
    risposte = {}
    out = np.empty((len(dati), len(STATI[nodo])))
    for k, riga in enumerate(dati.itertuples(index=False, name=None)):
        if riga not in risposte:
            q = inferenza.query([nodo], evidence=dict(zip(colonne, riga)), show_progress=False)
            ordine = [q.state_names[nodo].index(s) for s in STATI[nodo]]
            risposte[riga] = q.values[ordine]
        out[k] = risposte[riga]
    return out


def distribuzione_gravita(modello: DiscreteBayesianNetwork, df: pd.DataFrame,
                          colonne: list = OSSERVATE) -> np.ndarray:
    """P(gravita | colonne) per ogni riga, colonne nell'ordine di STATI['gravita']."""
    return _query(modello, df, colonne, "gravita")


def p_critico_righe(modello: DiscreteBayesianNetwork, df: pd.DataFrame,
                    colonne: list = OSSERVATE) -> np.ndarray:
    """P(gravita = critico | colonne) per ogni riga."""
    return distribuzione_gravita(modello, df, colonne)[:, STATI["gravita"].index("critico")]


def gravita_piu_probabile(modello: DiscreteBayesianNetwork, df: pd.DataFrame,
                          colonne: list = OSSERVATE) -> np.ndarray:
    """La rete come classificatore: lo stato di gravità con probabilità a posteriori massima."""
    p = distribuzione_gravita(modello, df, colonne)
    return np.array(STATI["gravita"])[p.argmax(axis=1)]


def velocita_piu_probabile(modello: DiscreteBayesianNetwork, df: pd.DataFrame,
                           colonne: list = OSSERVATE) -> np.ndarray:
    """Stima della variabile latente: la velocità del fuoco più probabile data l'evidenza."""
    p = _query(modello, df, colonne, "velocita_fuoco")
    return np.array(STATI["velocita_fuoco"])[p.argmax(axis=1)]


def scegli_alpha(df_stima: pd.DataFrame, df_evidenza: pd.DataFrame, colonne: list,
                 k: int = config.N_FOLD_CV, seme: int = config.SEME,
                 candidati: list = ALPHA_CANDIDATI) -> float:
    """Alpha di Laplace scelto in CV sui soli dati passati (macro F1 della rete come classificatore)."""
    from sklearn.metrics import f1_score

    y = df_stima["gravita"].astype(str).values
    punteggi = {}
    for alpha in candidati:
        f1 = []
        for tr, te in StratifiedKFold(k, shuffle=True, random_state=seme).split(df_stima, y):
            modello = stima(df_stima.iloc[tr], alpha)
            pred = gravita_piu_probabile(modello, df_evidenza.iloc[te], colonne)
            f1.append(f1_score(y[te], pred, average="macro"))
        punteggi[alpha] = float(np.mean(f1))
    return max(candidati, key=lambda a: (punteggi[a], -a))


def p_critico_fuori_fold(df_stima: pd.DataFrame, df_evidenza: pd.DataFrame,
                         k: int = config.N_FOLD_CV, seme: int = config.SEME,
                         colonne: list = OSSERVATE, alpha: float = config.LAPLACE_ALPHA) -> np.ndarray:
    """P(critico) per ogni riga di addestramento, senza usare la propria etichetta.

    Ogni parte è prevista da una rete stimata sulle altre k-1 parti. df_stima contiene i valori
    veri (per stimare le CPT); df_evidenza contiene le osservazioni che il sistema vede a runtime
    (per l'inferenza). Serve come feature per il ML sui dati di addestramento.
    """
    out = np.empty(len(df_stima))
    y = df_stima["gravita"].values
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seme).split(df_stima, y):
        modello = stima(df_stima.iloc[tr], alpha)
        out[te] = p_critico_righe(modello, df_evidenza.iloc[te], colonne)
    return out


def probabilita(modello: DiscreteBayesianNetwork, nodo: str, stato, **genitori) -> float:
    """P(nodo = stato | genitori) letta dalla CPT appresa."""
    cpd = modello.get_cpds(nodo)
    valori = {nodo: str(stato), **{k: str(v) for k, v in genitori.items()}}
    indice = tuple(cpd.state_names[v].index(valori[v]) for v in cpd.variables)
    return float(cpd.values[indice])
