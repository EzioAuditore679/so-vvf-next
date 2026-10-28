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
RUMORE_GRAVITA = 0.12            # probabilità che l'etichetta osservata si sposti di un livello (assunzione dichiarata)

# Ricerca nello spazio degli stati (dispatch dei mezzi)
# Le distanze tra stazioni e incidenti vengono da una griglia con ostacoli.
DIM_GRIGLIA = (20, 20)           # righe, colonne
PROB_OSTACOLO = 0.2              # frazione di celle bloccate
N_INCIDENTI_SCENARI = [2, 3, 4, 5, 6]  # dimensioni crescenti per il confronto BFS / UCS / A*
N_MEZZI_PER_INCIDENTE = 1.5      # mezzi disponibili = ceil(incidenti * questo valore)
N_SCENARI_PER_DIMENSIONE = 20    # ripetizioni per dimensione: si riportano media ± dev. std
LIMITE_NODI = 500_000            # oltre questo numero di nodi generati la ricerca si interrompe e il caso è segnato come non completato

# Percorsi
DATA_DIR = ROOT / "data"
KB_DIR = ROOT / "knowledge_base"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"
