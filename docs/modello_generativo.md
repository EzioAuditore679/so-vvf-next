# Modello generativo dei dati sintetici

Il dataset non è un campionamento indipendente delle variabili: ogni chiamata è prodotta da un
grafo causale esplicito, implementato in `src/modello_generativo.py`.

## Grafo causale

```
Tipo ──► Piano
 │  └──► Sostanze ──┐
 │                  ├──► Velocità del fuoco ◄── Meteo
 └──► Persone ◄── Piano
Persone, Velocità, Sostanze, Vulnerabilità, Adiacenti ──► Gravità  (+ rumore)
```

## Variabili

| Variabile | Osservata nel testo | Dipende da | Note |
|---|---|---|---|
| Tipo di edificio | sì | — | residenziale 0.40, industriale 0.35, bosco 0.25 |
| Piano | sì | Tipo | 0 per il bosco; pesi per 1–6 diversi per tipo |
| Sostanze pericolose | sì | Tipo | P(sì) = 0.30 residenziale, 0.75 industriale, 0.05 bosco |
| Meteo | sì | — | P(vento forte) = 0.40 |
| Velocità del fuoco | **no (nascosta)** | Meteo, Sostanze | tabella a 4 righe, vedi sotto |
| Persone | sì | Tipo, Piano | confermate crescono con il piano, tetto 0.65 |
| Vulnerabilità della zona | no, ma nota dal registro | — | P(alta) = 0.30; fatto del registro territoriale |
| Edifici adiacenti | no, ma noto dal registro | — | 0, 1, 2, 3 con 0.40, 0.30, 0.20, 0.10 |
| Gravità | no (target) | Persone, Velocità, Sostanze, Vulnerabilità, Adiacenti | regola più rumore |

### Velocità del fuoco, condizionata a meteo e sostanze

| Meteo | Sostanze | Alta | Media | Bassa |
|---|---|---|---|---|
| vento forte | sì | 0.70 | 0.25 | 0.05 |
| vento forte | no | 0.40 | 0.40 | 0.20 |
| calmo | sì | 0.35 | 0.40 | 0.25 |
| calmo | no | 0.10 | 0.30 | 0.60 |

## Regola di gravità

Punteggio = punti(persone) + punti(velocità) + [sostanze sì] + [vulnerabilità alta] + [adiacenti ≥ 2].

- Persone: confermate 2, incerte 1, assenti 0.
- Velocità: alta 2, media 1, bassa 0.
- Punteggio massimo 7.
- **Critico** se punteggio ≥ 5, **alto** se ≥ 3, altrimenti **basso**.

L'etichetta osservata è la regola con rumore: con probabilità `RUMORE_GRAVITA = 0.12` si tenta
uno spostamento di un livello, verso l'alto o verso il basso con uguale probabilità. Il rumore
rappresenta la variabilità non modellata del mondo reale.

Agli estremi lo spostamento verso l'esterno è troncato: da critico non si sale, da basso non si
scende, e l'etichetta resta invariata. Il tasso effettivo di etichette cambiate è quindi
`RUMORE_GRAVITA × (P(alto) + ½ P(critico) + ½ P(basso))`. Con la distribuzione delle classi del
generatore (basso 46%, alto 41%, critico 13%, misurata su 200 000 campioni) vale circa 0.705 × rumore:

| `RUMORE_GRAVITA` | tasso effettivo | tetto di accuratezza (1 − tasso) | accuratezza dell'oracolo (reg. logistica) |
|---|---|---|---|
| 0.05 | 0.035 | 0.965 | 0.974 ± 0.017 |
| 0.12 | 0.085 | 0.915 | 0.916 ± 0.021 |
| 0.20 | 0.141 | 0.859 | 0.847 ± 0.018 |

L'oracolo vede tutte le cause della gravità, quindi l'unico errore che non può eliminare è il rumore
dell'etichetta: la sua accuratezza coincide con il tetto teorico entro una deviazione standard.
È una verifica di coerenza tra generatore, dataset e pipeline di apprendimento
(valori da `results/sensitivita_rumore.csv`).

## Cosa vede ciascun modello

| Modello | Vede | Non vede |
|---|---|---|
| **Estrattore NLP** (adattatore) | testo della chiamata | tutto il resto |
| **ML, esperimento A** | tipo, piano, sostanze, persone, meteo (dal testo) | velocità, vulnerabilità, adiacenti |
| **ML, esperimento B** | A + vulnerabilità e adiacenti dal registro + P(critico) della rete bayesiana | velocità del fuoco |
| **ML, controllo C** | come B, senza la rete | velocità del fuoco |
| **ML, oracolo** | come C, più la velocità vera (non disponibile a runtime) | rumore dell'etichetta |
| **Rete bayesiana** | osservabili dal testo e vulnerabilità come evidenze; velocità del fuoco latente | adiacenti (non modellato, vedi limiti) |
| **Knowledge base Prolog** | osservabili, fatti del registro, velocità stimata o gravità stimata da una fonte esterna | rumore dell'etichetta |
| **Dispatch** (`valutazione_dispatch.py`) | la squadra derivata dalla KB con la gravità di una fonte | la gravità vera, usata solo per giudicare il piano |

## Regola anti-leakage

Il punteggio della regola di gravità **non va mai passato al modello ML come feature**: coincide
con il target a meno del rumore, e il modello apparirebbe perfetto senza imparare nulla.
Il ML riceve dal registro solo i fatti di base (vulnerabilità, adiacenti). Le CPT della rete bayesiana
sono stimate solo sui dati di training di ciascun fold di cross-validation, e P(critico) per le righe
di training è calcolata fuori fold (cross-fitting).

## Limiti da dichiarare

1. **Dati sintetici.** Le probabilità sono assunzioni scelte a mano, non stime da dati reali.
   Qualunque risultato dipende da queste scelte.
2. **Circolarità delle CPT.** La rete bayesiana stimata sui dati recupera le probabilità del
   generatore, che non è un esperto indipendente ma la fonte stessa dei dati. I test in
   `tests/test_bayes.py` verificano che con 20 000 righe lo stimatore le recuperi entro 0.03,
   non che siano corrette per il mondo reale.
3. **Adiacenti non modellato nella rete bayesiana.** La variabile è nel generatore e nella KB,
   ma non nella rete: il limite va riportato.
4. **Rumore arbitrario.** Il valore 0.12 è una scelta. L'analisi di sensibilità su 0.05, 0.12 e 0.20
   (`src/sensitivita.py`) mostra che le conclusioni non cambiano: B supera A, la rete è alla pari
   con C, l'oracolo resta molto sopra.
5. **Testo e campi osservabili.** Il testo è generato dai campi osservabili con template e
   rumore del parlato. Su questi testi l'estrattore NLP ricava tutti i campi correttamente, quindi
   nell'esperimento A i campi estratti coincidono con quelli veri: l'NLP non aggiunge rumore al ML.
   I test in `tests/test_nlp.py` verificano l'estrattore anche su frasi diverse dai template.
