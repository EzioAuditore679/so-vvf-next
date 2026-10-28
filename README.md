# SO-VVF-NEXT

Sistema basato su conoscenza per il dispatch dei mezzi in una sala operativa dei Vigili del Fuoco.
Progetto d'esame di Ingegneria della Conoscenza (ICON), Università di Bari. Dati interamente sintetici.

## Problema

Dato un insieme di incidenti attivi e una flotta di mezzi distribuiti in stazioni, scegliere quale mezzo assegnare a ciascun incidente:
- solo tra i mezzi **idonei** (regole Prolog su tipo di intervento, equipaggiamento e gravità);
- minimizzando il **tempo totale di percorrenza** (distanze da una griglia con ostacoli).

## Architettura

1. **Dati** (`src/generate_dataset.py`): chiamate strutturate con valori veri. Il modulo `src/nlp_extract.py` è un **adattatore di ingresso** che ricava i campi dal testo; non è oggetto di valutazione nella relazione.
2. **Knowledge base Prolog** (`knowledge_base/kb.pl`, interrogata via `pyswip`): idoneità dei mezzi, definizione della gravità a tre livelli, regole ricorsive (propagazione del pericolo tra strutture adiacenti), negazione per fallimento (`\+`) per incompatibilità tra mezzi e sostanze, abduzione sulle possibili cause.
3. **Ricerca nello spazio degli stati** (`src/search.py`):
   - **Stato**: insieme degli incidenti già assegnati, con il mezzo assegnato a ciascuno.
   - **Azione**: assegnare un mezzo idoneo e ancora libero a un incidente non ancora servito.
   - **Stato iniziale**: nessun incidente assegnato.
   - **Obiettivo**: tutti gli incidenti assegnati.
   - **Costo**: somma dei tempi di percorrenza.
   - **Algoritmi confrontati**: BFS e UCS (non informate) contro A* (informata).
4. **Euristica per A\*, da problema rilassato**: si rimuove l'ordine delle assegnazioni e si mantiene l'esclusività dei mezzi. Il costo ottimo dell'assegnamento dei rimanenti incidenti ai mezzi liberi (algoritmo ungherese, `scipy.optimize.linear_sum_assignment`) è un limite inferiore del costo reale, quindi h è **ammissibile**. È **consistente**: assegnare un incidente non può far calare h più del costo dell'azione stessa. Come confronto si usa anche una versione più debole, il minimo per incidente, da cui la sovrastima del numero di nodi espansi.
5. **Rete bayesiana** (`src/bayes_engine.py`): 8 nodi con CPT stimate per massima verosimiglianza con pseudoconteggi di Laplace. Produce la probabilità di gravità critica, usata come feature.
6. **Machine learning** (`src/train_models.py`): classificazione della gravità a tre livelli con **nested cross-validation** (esterna per la stima, interna per la scelta degli iperparametri). Esperimento A: feature grezze. Esperimento B: feature aggiunte dalla KB e dalla rete bayesiana. Risultati come media ± deviazione standard.

## Decisioni di progetto

- **Gravità a tre livelli** (critico, alto, basso), definita da regole Prolog dichiarate.
- **Nessun reasoner OWL.** La rappresentazione è relazionale in Prolog. La corrispondenza con una tassonomia di concetti è descritta nella documentazione, senza implementazione OWL.
- **CSP** formulato come assegnazione: variabili = incidenti, domini = mezzi idonei forniti da Prolog, vincolo = esclusività del mezzo. È risolto dalla ricerca nello spazio degli stati.
- **NLP** limitato all'adattatore di ingresso. Le sue prove restano in `results/` come controllo di funzionamento, non come risultato del progetto.
- **Esclusi**: clustering (non utile, i dati sono già classificati) e reti neurali (fuori dal nucleo del programma).

## Vincoli delle linee guida

- Nessuna matrice di confusione di un singolo run: le metriche ML sono medie ± deviazione standard da nested CV.
- Parametri documentati e motivati: pseudoconteggi di Laplace, numero di fold, dimensioni della griglia e degli scenari, in `config.py`.
- Complessità della KB misurata e riportata: numero di regole, predicati e profondità della ricorsione.

## Struttura

```
so-vvf-next/
├── README.md
├── requirements.txt
├── config.py                 # seme, dimensioni, soglie, percorsi
├── data/
│   ├── dataset_sintetico.csv # generato da generate_dataset.py
│   ├── dev_set.csv           # per tarare l'adattatore NLP
│   ├── gold_set.csv          # contaminato: usato per analisi errori
│   └── test_set.csv          # valutato una sola volta
├── knowledge_base/
│   └── kb.pl                 # regole Prolog (da scrivere)
├── results/                  # metriche, errori, storico
└── src/
    ├── domain.py             # vocabolario unico del dominio
    ├── generate_dataset.py   # generatore di chiamate sintetiche
    ├── nlp_extract.py        # adattatore di ingresso (regole e regex)
    ├── validate.py           # controllo di integrità e metriche NLP
    ├── prolog_engine.py      # interfaccia pyswip (da scrivere)
    ├── bayes_engine.py       # rete bayesiana (da scrivere)
    ├── search.py             # BFS, UCS, A* e euristica (da scrivere)
    ├── train_models.py       # nested CV, esperimenti A e B (da scrivere)
    └── main.py               # pipeline completa (da scrivere)
```

## Installazione

Richiede Python 3.10+ e SWI-Prolog installato nel sistema.

```
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python -c "import pyswip; print('ok')"
```

## Riproducibilità

```
python src/generate_dataset.py
python src/validate.py test
```

## Stato di avanzamento

- [x] Dominio, configurazione, generatore di dati
- [x] Adattatore NLP con dev, gold e test set
- [ ] Knowledge base Prolog
- [ ] Rete bayesiana
- [ ] Ricerca dispatch: BFS, UCS, A*, euristica
- [ ] Machine learning con nested CV
- [ ] Grafici e pipeline principale

Il gold set è contaminato dall'analisi degli errori. Il dettaglio è in `results/storico_estrattore.md`.
