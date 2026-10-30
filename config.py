"""Parametri di esecuzione per tutti i moduli"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Valore del seme
SEME = 42

# Dataset
N_TESTI_SINTETICI = 1000
N_FOLD_CV = 5
RUMORE_GRAVITA = 0.12  # probabilità che l'etichetta osservata si sposti di un livello. Assunzione dichiarata.
LAPLACE_ALPHA = 1.0  # pseudoconteggio per cella delle CPT della rete bayesiana

# Ricerca nello spazio degli stati (dispatch dei mezzi)
DIM_GRIGLIA = (20, 20)
PROB_OSTACOLO = 0.2
N_INCIDENTI_SCENARI = [2, 4, 6, 8, 10]
# Un incidente richiede in media 1.6 ruoli (misurato sul modello generativo, docs/ricerca.md §2.1).
N_MEZZI_PER_INCIDENTE = 2.2  # mezzi disponibili = ceiling(incidenti * questo valore)
# Composizione della flotta, proporzionale alla domanda media di ciascun ruolo.
COMPOSIZIONE_FLOTTA = {"autopompa": 0.50, "nucleo_nbcr": 0.20, "canadair": 0.15, "autoscala": 0.15}
# Costo di un ruolo lasciato scoperto (rinforzi da un comando vicino). Supera il tempo di percorrenza
# tipico sulla griglia, quindi la ricerca lascia scoperto un ruolo solo se non ha alternative.
PENALITA_RUOLO_SCOPERTO = 60
DISTANZA_ADIACENZA = 3  # due incidenti a distanza di Manhattan <= 3 sono adiacenti
N_SCENARI_PER_DIMENSIONE = 20  # si riportano media ± deviazione standard
LIMITE_NODI = 500_000  # oltre questo numero di nodi il caso è segnato come non completato

# Valutazione end-to-end delle fonti di gravità sul dispatch
N_INCIDENTI_E2E = 5
N_SCENARI_E2E = 100
SEME_E2E = SEME + 1000  # scenari indipendenti dal dataset di addestramento

# Percorsi
DATA_DIR = ROOT / "data"
KB_DIR = ROOT / "knowledge_base"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"
