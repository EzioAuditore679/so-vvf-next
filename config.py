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
LAPLACE_ALPHA = 1.0  # pseudoconteggio per cella delle CPT della rete bayesiana

# Ricerca nello spazio degli stati (dispatch dei mezzi). Distanze da griglia con ostacoli.
DIM_GRIGLIA = (20, 20)  # righe, colonne
PROB_OSTACOLO = 0.2
N_INCIDENTI_SCENARI = [2, 4, 6, 8, 10]
# Un incidente richiede in media 1.6 ruoli (misurato sul modello generativo, docs/ricerca.md §2.1).
N_MEZZI_PER_INCIDENTE = 2.2  # mezzi disponibili = ceil(incidenti * questo valore)
# Composizione della flotta, proporzionale alla domanda media di ciascun ruolo.
COMPOSIZIONE_FLOTTA = {"autopompa": 0.50, "nucleo_nbcr": 0.20, "canadair": 0.15, "autoscala": 0.15}
# Costo di un ruolo lasciato scoperto (rinforzi da un comando vicino). Supera il tempo di percorrenza
# tipico sulla griglia 20x20, quindi la ricerca lascia scoperto un ruolo solo se non ha alternative.
PENALITA_RUOLO_SCOPERTO = 60
DISTANZA_ADIACENZA = 3  # due incidenti a distanza di Manhattan <= 3 sono adiacenti (propagazione)
N_SCENARI_PER_DIMENSIONE = 20  # si riportano media ± deviazione standard
LIMITE_NODI = 500_000  # oltre questo numero di nodi il caso è segnato come non completato

# Percorsi
DATA_DIR = ROOT / "data"
KB_DIR = ROOT / "knowledge_base"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"
