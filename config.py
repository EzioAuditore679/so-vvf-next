"""Parametri di esecuzione di SO-VVF-NEXT.

Tutti i risultati devono essere riproducibili: il seme e le dimensioni
vanno cambiati qui, non nei singoli moduli.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Riproducibilità
SEME = 42

# Dataset
N_TESTI_SINTETICI = 1000
N_GOLD_SET = 100                 # etichettati a mano, mai usati in addestramento
QUOTA_TEST = 0.2                 # frazione del dataset sintetico riservata al test
N_FOLD_CV = 5                    # stratified k-fold per il ML

# Griglia di ricerca
DIM_GRIGLIA = (20, 20)           # righe, colonne
PROB_OSTACOLO = 0.2              # frazione di celle bloccate
N_ISTANZE_RICERCA = 30           # coppie partenza/arrivo per il confronto
SOGLIA_COPERTURA = 15.0          # tempo massimo di raggiungibilità di un settore (unità di tempo)

# Percorsi
DATA_DIR = ROOT / "data"
KB_DIR = ROOT / "knowledge_base"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"
