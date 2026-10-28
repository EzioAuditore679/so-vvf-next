# Storico delle valutazioni dell'estrattore NLP

Regola del progetto: le regole di `nlp_extract.py` si tarano solo sul **dev set**.
Il **test set** si valuta una volta sola. Il **gold set** è stato usato per analizzare gli errori
e quindi non è più un test indipendente.

## Versione 1 (regole iniziali)

Valutata su dataset sintetico (ottimistico) e poi su gold set (primo test indipendente).

| Campo | Gold (v1) accuratezza | Gold (v1) F1 macro |
|---|---|---|
| Piano | 1.00 | n.d. |
| Tipo di edificio | 0.74 | 0.84 |
| Sostanze pericolose | 0.93 | 0.93 |
| Persone | 0.76 | 0.82 |
| Meteo | 0.77 | 0.84 |

Baseline sul dev set (v1): tipo di edificio 0.57, sostanze 0.90, persone 0.67, meteo 0.70.

## Versione 2 (dopo correzioni sul dev set)

Correzioni: sinonimi per edifici (casa, palazzo, villetta, officina, magazzino, fabbrica),
sostanze (benzina, GPL, idrogeno, ossigeno, cloro, propano, tanica), negazioni ("non c'è / non ci sono"),
assenze e incertezza espresse in più modi, meteo calmo ("aria ferma", "tempo stabile", "non tira vento").

| Insieme | Tipo di edificio | Sostanze | Persone | Meteo | Note |
|---|---|---|---|---|---|
| Dev (usato per tarare) | 1.00 | 1.00 | 1.00 | 1.00 | Non è un test: è l'insieme di correzione |
| Gold (contaminato) | 0.99 | 0.98 | 0.98 | 1.00 | Gli errori del gold hanno guidato le parole aggiunte: non indipendente |
| **Test (una sola valutazione)** | **1.00** | **1.00** | **0.97** | **1.00** | 1 errore su 30 testi: "Non ci sono persone dentro" → non riconosciuto |

## Limiti da dichiarare nella documentazione

- Tutti i testi (dev, gold, test) sono stati scritti dallo stesso autore: l'assistente. Il test set
  non è quindi indipendente dal modo di scrivere delle regole. Un test vero richiederebbe testi scritti
  da persone diverse, ad esempio colleghi o compagni di corso.
- Il test set contiene 30 testi: è piccolo, e ogni errore sposta le metriche di circa 3 punti.
- Il risultato sul test set non è stato usato per correggere la regola residua (persone assenti).
