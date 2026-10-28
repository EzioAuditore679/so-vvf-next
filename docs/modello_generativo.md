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
| Vulnerabilità della zona | **no (nascosta)** | — | P(alta) = 0.30; fatto del registro territoriale |
| Edifici adiacenti | **no (nascosta)** | — | 0, 1, 2, 3 con 0.40, 0.30, 0.20, 0.10 |
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

L'etichetta osservata è la regola con rumore: con probabilità `RUMORE_GRAVITA = 0.12` si sposta
di un livello (verso l'alto o verso il basso). Il rumore rappresenta la variabilità non
modellata del mondo reale.

## Cosa vede ciascun modello

| Modello | Vede | Non vede |
|---|---|---|
| **Estrattore NLP** (adattatore) | testo della chiamata | tutto il resto |
| **ML, esperimento A** | tipo, piano, sostanze, persone, meteo (dal testo) | velocità, vulnerabilità, adiacenti |
| **ML, esperimento B** | A + vulnerabilità e adiacenti dalla KB + probabilità della rete bayesiana | velocità del fuoco |
| **Rete bayesiana** | osservabili come evidenze; velocità e vulnerabilità come latenti | adiacenti (non modellato, vedi limiti) |
| **Knowledge base Prolog** | fatti del registro (vulnerabilità, adiacenti) e osservabili | rumore dell'etichetta |

## Regola anti-leakage

Il punteggio della regola di gravità **non va mai passato al modello ML come feature**: coincide
con il target a meno del rumore, e il modello apparirebbe perfetto senza imparare nulla.
La KB fornisce al ML solo i fatti di base (vulnerabilità, adiacenti). Le CPT della rete bayesiana
sono stimate solo sui dati di training di ciascun fold di cross-validation.

## Limiti da dichiarare

1. **Dati sintetici.** Le probabilità sono assunzioni scelte a mano, non stime da dati reali.
   Qualunque risultato dipende da queste scelte.
2. **Circolarità delle CPT.** La rete bayesiana stimata sui dati recupera le probabilità del
   generatore. Per questo si confrontano le CPT apprese con CPT definite da esperti (analisi di sensitività).
3. **Adiacenti non modellato nella rete bayesiana.** La variabile è nel generatore e nella KB,
   ma non nella rete: il limite va riportato.
4. **Rumore arbitrario.** Il valore 0.12 è una scelta. La sensitività su 0.05, 0.12 e 0.20
   mostra quanto il risultato dipende dal rumore.
5. **Testo e campi osservabili.** Il testo è generato dai campi osservabili con template e
   rumore del parlato: l'estrattore NLP lavora su testi che il generatore ha costruito, quindi i suoi
   risultati sul dataset sintetico restano ottimistici (vedi `results/storico_estrattore.md`).
