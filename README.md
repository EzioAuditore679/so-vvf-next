# SO-VVF-NEXT

Sistema basato su conoscenza per il dispatch dei mezzi in una sala operativa dei Vigili del Fuoco.
Progetto d'esame di Ingegneria della Conoscenza (ICON), Università di Bari. Dati interamente sintetici.

## Problema

Dato un insieme di incidenti attivi e una flotta di mezzi distribuiti in stazioni, scegliere quale mezzo assegnare a ciascun incidente:
- solo tra i mezzi **idonei** (regole Prolog su tipo di intervento, equipaggiamento e gravità);
- minimizzando il **tempo totale di percorrenza**, calcolato su una griglia con ostacoli.

## Architettura

1. **Dati** (`src/modello_generativo.py`, `src/generate_dataset.py`): un modello causale esplicito genera la situazione (tipo di edificio, piano, sostanze, meteo, velocità del fuoco, persone, vulnerabilità della zona, edifici adiacenti) e la gravità, con rumore dichiarato. Il testo della chiamata è generato solo dai campi osservabili. Dettagli in `docs/modello_generativo.md`.
   `src/nlp_extract.py` è un **adattatore di ingresso** che ricava i campi dal testo; non è oggetto di valutazione nella relazione.

2. **Knowledge base Prolog** (`knowledge_base/kb.pl`, interrogata via `pyswip` da `src/prolog_engine.py`): idoneità dei mezzi con negazione per fallimento (`\+`) per l'incompatibilità canadair–sostanze pericolose; regola di gravità a tre livelli, identica a `regola_gravita` (verificata da test su 2000 casi); abduzione sulla velocità del fuoco (la variabile latente che spiega la gravità osservata); propagazione ricorsiva del pericolo tra edifici adiacenti, con lista dei visitati per evitare i cicli; vincoli di integrità sulle proposte di dispatch.
   - Soglie e punteggi della gravità non sono scritti a mano in Prolog: `src/prolog_export.py` li genera da `modello_generativo.py` in `knowledge_base/kb_statica.pl`, che `kb.pl` include. Un test verifica che il file generato sia aggiornato.
   - Complessità di `kb.pl` misurata con `KnowledgeBase().complessita()`: 16 predicati, 21 regole con corpo, 25 clausole. La profondità della ricorsione di `collegato/2` è al massimo il numero di incidenti dello scenario.
   - I fatti di scenario (incidenti, mezzi, registro, velocità) vengono asseriti da Python per ogni scenario. Le regole restano fisse nel file.

3. **Ricerca a due livelli**:
   - **Livello 1, distanze** (`src/search.py`): A\* sulla griglia 20×20 con ostacoli calcola il tempo minimo $T(m, i)$ tra ogni mezzo e ogni incidente. Euristica $h = d_{\text{euclidea}} / V_{\max}$, con $V_{\max}$ maggiore o uguale alla velocità di ogni arco, quindi ammissibile. Il costo è verificato contro Dijkstra su tutti i casi di test.
   - **Livello 2, dispatch** (stesso modulo): stato = assegnazioni incidente-mezzo già fatte; azione = assegnare un mezzo idoneo e libero a un incidente non servito; obiettivo = tutti gli incidenti serviti; costo = somma dei tempi $T(m, i)$. Algoritmi confrontati: **BFS** e **UCS** (non informate) contro **A\*** con due euristiche:
     - $h_{\min}$ = somma, per ogni incidente rimasto, del minimo tempo verso un mezzo libero. Ammissibile, più debole.
     - $h_{\text{Hungarian}}$ = costo ottimo dell'assegnamento dei rimanenti incidenti ai mezzi liberi (`scipy.optimize.linear_sum_assignment`). Ottenuta rimuovendo l'ordine delle assegnazioni e mantenendo l'esclusività dei mezzi, è un limite inferiore più stretto: domina $h_{\min}$, quindi espande meno nodi a parità di ammissibilità.
     - Entrambe ammissibili e consistenti. I risultati sono riportati come media ± deviazione standard su più scenari per ogni dimensione.

4. **Rete bayesiana** (`src/bayes_engine.py`): 8 nodi: Tipo, Piano, Sostanze, Meteo, Velocità del fuoco, Persone, Vulnerabilità, Gravità. Velocità e vulnerabilità sono latenti per il modello di inferenza. CPT stimate per massima verosimiglianza con pseudoconteggi di Laplace. Produce la probabilità di gravità critica, calcolata dentro ogni fold.

5. **Machine learning** (`src/train_models.py`): classificazione della gravità a tre livelli con **nested cross-validation** (esterna per la stima, interna per la scelta degli iperparametri).
   - Esperimento A: solo i campi osservabili dal testo.
   - Esperimento B: A + fatti della KB (vulnerabilità, adiacenti) + probabilità della rete bayesiana.
   - Il punteggio della regola di gravità **non entra mai nelle feature** (target leakage).

## Decisioni di progetto

- **Gravità a tre livelli** (critico, alto, basso), definita da una regola esplicita, scritta in Prolog.
- **Nessun reasoner OWL e nessun esportatore OWL.** La rappresentazione è relazionale in Prolog. La corrispondenza con una tassonomia di concetti è descritta nella documentazione, senza implementazione OWL.
- **CSP** formulato come assegnazione: variabili = incidenti, domini = mezzi idonei forniti da Prolog, vincolo = esclusività del mezzo. È risolto dalla ricerca nello spazio degli stati.
- **Dati generati da un modello causale**, non da campionamento indipendente. Le probabilità sono assunzioni dichiarate, con analisi di sensitività.
- **NLP** limitato all'adattatore di ingresso. Le sue prove restano in `results/` come controllo di funzionamento.
- **Esclusi**: clustering (non utile, i dati sono già classificati) e reti neurali (fuori dal nucleo del programma).

## Vincoli delle linee guida

- Nessuna matrice di confusione di un singolo run: le metriche ML sono medie ± deviazione standard da nested CV.
- Parametri documentati e motivati, in `config.py` e in `docs/`: pseudoconteggi di Laplace, numero di fold, dimensioni della griglia e degli scenari, probabilità e rumore del modello generativo.
- Complessità della KB misurata e riportata: numero di regole, predicati e profondità della ricorsione.

## Struttura

```
so-vvf-next/
├── README.md
├── requirements.txt
├── config.py                  # seme, dimensioni, rumore, percorsi
├── docs/
│   └── modello_generativo.md  # grafo causale, probabilità, limiti
├── data/
│   ├── dataset_sintetico.csv  # generato da generate_dataset.py
│   ├── dev_set.csv            # per tarare l'adattatore NLP
│   ├── gold_set.csv           # contaminato: usato per analisi errori
│   └── test_set.csv           # valutato una sola volta
├── knowledge_base/
│   ├── kb.pl                  # regole Prolog
│   └── kb_statica.pl          # soglie e punteggi, generato da prolog_export.py
├── results/                   # metriche, errori, storico
├── tests/
│   └── test_prolog.py         # coerenza KB/regola Python, idoneità, abduzione, ricorsione, integrità
└── src/
    ├── domain.py              # vocabolario unico del dominio
    ├── modello_generativo.py  # grafo causale e regola di gravità
    ├── generate_dataset.py    # generatore di chiamate e stato del mondo
    ├── nlp_extract.py         # adattatore di ingresso (regole e regex)
    ├── validate.py            # controllo di integrità e metriche NLP
    ├── prolog_export.py       # genera kb_statica.pl dalle costanti Python
    ├── prolog_engine.py       # interfaccia pyswip verso kb.pl
    ├── bayes_engine.py        # rete bayesiana (da scrivere)
    ├── search.py              # A* spaziale, dispatch: BFS, UCS, A* (da scrivere)
    ├── train_models.py        # nested CV, esperimenti A e B (da scrivere)
    └── main.py                # pipeline completa (da scrivere)
```

## Installazione

Richiede Python 3.10+ e SWI-Prolog installato nel sistema (su Ubuntu/Debian: `sudo apt-get install swi-prolog`; su Windows, installer ufficiale e `swipl` nel PATH).

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
python -m pytest tests/        # include il controllo di coerenza tra KB Prolog e regola Python
```

## Stato di avanzamento

- [x] Dominio, configurazione, modello generativo causale, generatore di dati
- [x] Adattatore NLP con dev, gold e test set
- [x] Knowledge base Prolog, con test di coerenza con `regola_gravita` (`tests/test_prolog.py`)
- [ ] Rete bayesiana
- [ ] Ricerca: A* spaziale, dispatch con BFS, UCS, A* ($h_{\min}$ e $h_{\text{Hungarian}}$)
- [ ] Machine learning con nested CV, esperimenti A e B
- [ ] Analisi di sensitività (CPT apprese contro esperti, rumore 0.05 / 0.12 / 0.20)
- [ ] Grafici e pipeline principale

Il gold set è contaminato dall'analisi degli errori. Il dettaglio è in `results/storico_estrattore.md`.
