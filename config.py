"""Parametri di esecuzione. Seme, dimensioni e rumore si cambiano qui, non nei singoli moduli."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

SEME = 42

# Dataset
N_TESTI_SINTETICI = 1000
N_GOLD_SET = 100
QUOTA_TEST = 0.2
N_FOLD_CV = 5
RUMORE_GRAVITA = 0.12  # probabilità che l'etichetta osservata si sposti di un livello. Assunzione dichiarata.

# Ricerca nello spazio degli stati (dispatch dei mezzi). Distanze da griglia con ostacoli.
DIM_GRIGLIA = (20, 20)  # righe, colonne
PROB_OSTACOLO = 0.2
N_INCIDENTI_SCENARI = [2, 3, 4, 5, 6]
N_MEZZI_PER_INCIDENTE = 1.5  # mezzi disponibili = ceil(incidenti * questo valore)
N_SCENARI_PER_DIMENSIONE = 20  # si riportano media ± deviazione standard
LIMITE_NODI = 500_000  # oltre questo numero di nodi il caso è segnato come non completato

# Percorsi
DATA_DIR = ROOT / "data"
KB_DIR = ROOT / "knowledge_base"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"
