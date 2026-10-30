# SO-VVF-NEXT

Sistema basato su conoscenza per il dispatch dei mezzi in una sala operativa dei Vigili del Fuoco.
Progetto d'esame di Ingegneria della Conoscenza (ICON), Università di Bari. Dati interamente sintetici.

## Problema

Dato un insieme di chiamate al 115 e una flotta di mezzi distribuiti sul territorio, decidere quali mezzi inviare a ciascun incidente:
- la **gravità** della chiamata è stimata (ML, rete bayesiana, oppure rete + regola della KB), perché la velocità del fuoco non si osserva;
- la **squadra richiesta** (estinzione, rinforzo, soccorso in quota, decontaminazione) è derivata dalla KB Prolog a partire dalla gravità operativa, che tiene conto della propagazione tra incidenti vicini;
- i mezzi sono assegnati ai ruoli minimizzando il **tempo totale di percorrenza** su una griglia con ostacoli, con una penalità per i ruoli lasciati scoperti.

La catena completa è: testo della chiamata → campi → gravità stimata → squadra (KB) → dispatch (A*).

## Architettura

1. **Dati** (`src/modello_generativo.py`, `src/generate_dataset.py`): un modello causale esplicito genera la situazione (tipo di edificio, piano, sostanze, meteo, velocità del fuoco, persone, vulnerabilità della zona, edifici adiacenti) e la gravità, con rumore dichiarato. Il testo della chiamata è generato solo dai campi osservabili. Dettagli in `docs/modello_generativo.md`.
   `src/nlp_extract.py` è un **adattatore di ingresso** che ricava i campi dal testo; non è oggetto di valutazione sperimentale ed è verificato da test automatici (`tests/test_nlp.py`).

2. **Knowledge base Prolog** (`knowledge_base/kb.pl`, interrogata via `pyswip` da `src/prolog_engine.py`): ruoli della squadra derivati dalla gravità operativa e mezzi adatti a ogni ruolo, con negazione per fallimento (`\+`) per l'incompatibilità canadair–sostanze pericolose; spiegazioni dei ruoli (`motivo/3`); gravità stimata da una fonte esterna che prevale sulla regola; regola di gravità a tre livelli, identica a `regola_gravita` (verificata da test su 2000 casi); abduzione sulla velocità del fuoco (la variabile latente che spiega la gravità osservata); propagazione ricorsiva del pericolo tra edifici adiacenti, con lista dei visitati per evitare i cicli; vincoli di integrità sulle proposte di dispatch.
   - La KB è un solo file: in cima la conoscenza di dominio (punti e soglie della gravità, soglia dei piani per l'autoscala), poi le regole. I fatti di dominio coincidono con le costanti del generatore Python: lo verificano un test diretto sulle costanti e un test sulla gravità di 2000 casi.
   - Complessità di `kb.pl` misurata con `KnowledgeBase().complessita()`: 29 predicati, 37 regole con corpo, 52 clausole (di cui 9 fatti di dominio). La profondità della ricorsione di `collegato/2` è al massimo il numero di incidenti dello scenario.
   - I fatti di scenario (incidenti, mezzi, registro, velocità) vengono asseriti da Python per ogni scenario. Le regole restano fisse nel file.

3. **Ricerca a due livelli** (`src/search.py`, dimostrazioni in `docs/ricerca.md`):
   - **Livello 1, distanze**: sulla griglia $20 \times 20$ con ostacoli, A\* con euristica di Manhattan calcola la distanza tra ogni stazione e ogni incidente. Manhattan è la soluzione esatta del problema senza ostacoli, quindi ammissibile e consistente. Verificata contro le distanze esatte.
   - **Livello 2, dispatch**: CSP risolto con la ricerca. Una variabile per ruolo richiesto, in ordine MRV; stato = (ruoli decisi, mezzi occupati); azione = un mezzo adatto e libero, costo $\max(T(m,i), 1)$ con $\varepsilon = 1$ (Proposizione 2.1), oppure ruolo scoperto con penalità fissa. Algoritmi: **LcFS** (ricerca a costo uniforme, non informata) contro **A\*** con due euristiche da problemi semplificati:
     - $h_{\min}$: si rimuove l'esclusività dei mezzi; minimo statico per ruolo.
     - $h_{\text{libero}}$: esclusi solo i mezzi già occupati.
     - Entrambe ammissibili e consistenti (dimostrazioni nella documentazione). I risultati sono medie ± deviazione standard su 20 scenari per dimensione.

4. **Rete bayesiana** (`src/bayes_engine.py`): 8 nodi: Tipo, Piano, Sostanze, Meteo, Velocità del fuoco, Persone, Vulnerabilità, Gravità. CPT stimate per massima verosimiglianza con pseudoconteggi di Laplace (alpha scelto in CV interna). La velocità del fuoco è latente. Usi: P(critico) come feature, classificatore della gravità, stima della velocità del fuoco da passare alla regola della KB.

5. **Machine learning** (`src/train_models.py`): classificazione della gravità a tre livelli con **nested cross-validation** (esterna per la stima, interna per la scelta degli iperparametri).
   - Esperimento A: solo i campi osservabili dal testo.
   - Esperimento B: A + fatti del registro (vulnerabilità, adiacenti) + probabilità della rete bayesiana, con la vulnerabilità come evidenza.
   - Rete: la rete bayesiana come classificatore. Controlli: C = B senza rete; oracolo = C con la velocità vera.
   - Il punteggio della regola di gravità **non entra mai nelle feature** (target leakage).

6. **Valutazione end-to-end** (`src/valutazione_dispatch.py`): su 100 scenari la squadra e il dispatch sono calcolati con la gravità di ogni fonte e giudicati con la gravità vera (ruoli scoperti, mezzi superflui, critici mancati, costo in più rispetto all'ottimo).

## Decisioni di progetto

- **Gravità a tre livelli** (critico, alto, basso), definita da una regola esplicita, scritta in Prolog.
- **Nessun reasoner OWL e nessun esportatore OWL.** La rappresentazione è relazionale in Prolog: le regole del dominio (squadra, propagazione, abduzione, integrità) richiedono ricorsione, negazione per fallimento e aritmetica, mentre una tassonomia di concetti non aggiungerebbe ragionamento utile al problema. La motivazione è nella relazione.
- **CSP** formulato come assegnazione: variabili = ruoli richiesti dagli incidenti, domini = mezzi adatti forniti da Prolog più il valore *scoperto*, vincolo = esclusività del mezzo. È risolto dalla ricerca nello spazio degli stati, con ordine MRV delle variabili.
- **Dati generati da un modello causale**, non da campionamento indipendente. Le probabilità sono assunzioni dichiarate, con analisi di sensibilità.
- **NLP** limitato all'adattatore di ingresso, verificato da test su sinonimi, negazioni, incertezza e abbreviazioni e su tutte le chiamate del generatore. Un solo dataset, come per gli altri componenti: niente insiemi di testi scritti a mano da valutare.
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
│   ├── modello_generativo.md  # grafo causale, probabilità, limiti
│   └── ricerca.md             # rappresentazione, euristiche, dimostrazioni, risultati
├── data/
│   └── dataset_sintetico.csv  # generato da generate_dataset.py (seme 42)
├── knowledge_base/
│   └── kb.pl                  # la KB: conoscenza di dominio e regole
├── results/                   # risultati in CSV
├── plots/                     # grafici generati da plotter.py
├── tests/
│   ├── test_prolog.py         # coerenza KB/regola Python, squadra, abduzione, ricorsione, integrità
│   ├── test_search.py         # ammissibilità, consistenza, ottimalità delle euristiche
│   ├── test_bayes.py          # CPT, Laplace, inferenza, recupero del generatore
│   ├── test_nlp.py            # regole dell'adattatore NLP e lettura delle chiamate generate
│   └── test_valutazione.py    # giudizio dei piani con la gravità vera
└── src/
    ├── domain.py              # vocabolario unico del dominio
    ├── modello_generativo.py  # grafo causale e regola di gravità
    ├── generate_dataset.py    # generatore di chiamate e stato del mondo
    ├── nlp_extract.py         # adattatore di ingresso (regole e regex)
    ├── prolog_engine.py       # interfaccia pyswip verso kb.pl
    ├── bayes_engine.py        # rete bayesiana: stima, inferenza, cross-fitting
    ├── search.py              # ricerca su griglia e dispatch per ruoli: LcFS, A*
    ├── train_models.py        # nested CV, esperimenti A, B, rete, C, oracolo
    ├── sensitivita.py         # recupero delle CPT e rumore delle etichette
    ├── valutazione_dispatch.py # fonti di gravità valutate sul dispatch
    ├── plotter.py             # grafici dai CSV in results/
    └── main.py                # demo integrata e pipeline degli esperimenti
```

## Installazione

Richiede Python 3.10+ e SWI-Prolog installato nel sistema (su Ubuntu/Debian: `sudo apt-get install swi-prolog`; su Windows, installer ufficiale e `swipl` nel PATH).

```
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python -c "import pyswip; print('ok')"
```

## Uso

```
python src/main.py                                  # demo: 4 chiamate dal testo al dispatch, con spiegazioni (pochi secondi)
python src/main.py esperimenti                      # rigenera dataset, test, risultati e grafici (~15 minuti)
python src/main.py esperimenti --senza-sensibilita  # come sopra, senza l'analisi di sensibilità (~5 minuti)
python src/main.py grafici                          # solo i grafici in plots/, dai CSV in results/ (src/plotter.py)
```

La demo stima la gravità con ML, rete bayesiana e rete + regola della KB, ricava la squadra dalla KB con
le motivazioni di ogni ruolo, assegna i mezzi con A* e verifica la proposta con i vincoli di integrità in Prolog.
Il dispatch usa la gravità della fonte rete + KB, che nella valutazione end-to-end manca meno incidenti critici.

Grafici in `plots/`, prodotti da `src/plotter.py`: `ml_modelli.png` (accuratezza, F1 macro e F1 pesata, media ± dev. std. per esperimento e modello), `ricerca_espansi.png`
(nodi espansi di LcFS e A* al crescere degli incidenti), `dispatch_fonti.png` (fonti della gravità valutate sul dispatch).

## Riproducibilità dei singoli passi

```
python src/generate_dataset.py
python -m pytest tests/        # KB, ricerca, rete bayesiana, adattatore NLP, valutazione
python src/search.py           # confronto LcFS e A* sul dispatch
python src/train_models.py     # nested CV
python src/sensitivita.py      # analisi di sensibilità
python src/valutazione_dispatch.py  # fonti di gravità sul dispatch
```

## Stato di avanzamento

- [x] Dominio, configurazione, modello generativo causale, generatore di dati
- [x] Adattatore NLP, verificato da test automatici
- [x] Knowledge base Prolog, con test di coerenza con `regola_gravita` (`tests/test_prolog.py`)
- [x] Rete bayesiana, con test
- [x] Ricerca: distanze A* su griglia, dispatch per ruoli con LcFS e A* ($h_{\min}$ e $h_{\text{libero}}$), test e confronto
- [x] Machine learning con nested CV: esperimenti A, B, rete, controlli C e oracolo
- [x] Analisi di sensibilità: recupero delle CPT rispetto al generatore, rumore 0.05 / 0.12 / 0.20
- [x] Valutazione end-to-end delle fonti di gravità sul dispatch
- [x] Pipeline principale, demo e grafici
- [ ] Relazione
