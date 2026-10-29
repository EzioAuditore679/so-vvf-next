# Ricerca nello spazio degli stati

Documento di riferimento per `src/search.py` e `tests/test_search.py`. Segue il capitolo
"Ricerca in Spazi di Stati" del corso (sezioni 2.2–2.8, Proposizioni 2.1 e 2.2, Tabella 2.1).

Il problema è risolto su due livelli. Il livello 1 calcola le distanze tra stazioni e incidenti.
Il livello 2 decide quale mezzo va a quale incidente, usando quelle distanze come costi.

---

## 1. Livello 1: distanze sulla griglia

### 1.1 Rappresentazione

Il modello è quello dei videogiochi (appunti, Es. 2.3) e della mappa stradale (Es. 2.1, 2.2):

| Elemento | Definizione |
|---|---|
| Stati | celle libere $(r, c)$ di una griglia $20 \times 20$ |
| Stato iniziale | cella della stazione del mezzo $m$ |
| Azioni | spostamento di una cella nelle 4 direzioni, verso una cella libera |
| Costo di un arco | $\text{cost}(n, n') = 1$ per ogni passo |
| Obiettivo | $\text{goal}(s) = (s = \text{cella dell'incidente } i)$ |
| Soluzione | cammino di celle dalla stazione all'incidente, di costo = numero di passi |

Il grafo è implicito: archi e nodi sono generati da `Griglia.successori`. Gli ostacoli sono una frazione
`config.PROB_OSTACOLO` delle celle. L'unità di costo è il passo di griglia, quindi il tempo di percorrenza
è il numero di passi divisi per la velocità, che è 1 cella per unità di tempo.

### 1.2 Problema semplificato ed euristica

**Problema semplificato:** la stessa griglia senza ostacoli, con gli stessi movimenti a 4 direzioni.
La sua soluzione ottima da $n = (r, c)$ a $g = (r_g, c_g)$ è la distanza di Manhattan:

$$h(n) = |r - r_g| + |c - c_g|$$

Il calcolo è in forma chiusa, quindi il rilassamento è risolto esattamente.

**Scelta rispetto agli appunti.** Gli appunti (Es. 2.16) propongono la distanza euclidea. La scelta di Manhattan
è più stretta, perché è la soluzione esatta del problema semplificato con i nostri movimenti. La euclidea
resta ammissibile ma sottostima di più il costo reale.

### 1.3 Ammissibilità e monotonicità

**Ammissibilità.** Ogni cammino da $n$ a $g$ fa almeno $|r - r_g| + |c - c_g|$ passi, perché ogni passo cambia
una sola coordinata di 1. Quindi $h(n) \le h^*(n)$.

**Monotonicità (consistenza).** Per ogni arco $n \to n'$ con costo 1, $h$ cambia di al più 1, quindi

$$h(n) \le \text{cost}(n, n') + h(n')$$

La verifica è in `test_manhattan_ammissibile_e_monotona_sulla_griglia`, su griglie casuali e confrontando $h$
con le distanze esatte calcolate da una BFS a ritroso.

---

## 2. Livello 2: assegnazione dei mezzi

### 2.1 Rappresentazione

| Elemento | Definizione |
|---|---|
| Stati | insiemi $s = \{(m_1, i_1), \ldots, (m_k, i_k)\}$ di assegnazioni, con ogni $m$ e ogni $i$ al più una volta |
| Stato iniziale | $s_0 = \emptyset$ |
| Azioni | assegnare un incidente $i \notin s$ a un mezzo $m$ idoneo e libero |
| Transizione | $s' = s \cup \{(m, i)\}$ |
| Costo di un'azione | $C(m, i) = \max(T(m, i), 1)$, con $T$ la distanza del livello 1 |
| Obiettivo | tutti gli incidenti sono serviti |
| Soluzione | insieme di assegnazioni di costo minimo |

**Idoneità.** Quali mezzi possono servire un incidente viene dalla knowledge base Prolog
(`idoneo/2` in `knowledge_base/kb.pl`), non dal codice Python.

**Stati come insiemi.** Ordini diversi della stessa assegnazione producono lo stesso stato. Per questo lo stato
è un `frozenset`, e la lista dei chiusi riconosce i duplicati (potatura dei cammini multipli, §2.7.2 del corso).
Senza questa scelta lo spazio si gonfia per permutazioni.

**Dimensione dello spazio di stati.** Con $n$ incidenti e $m$ mezzi, il numero di stati raggiungibili è

$$\sum_{k=0}^{n} \binom{n}{k} \cdot \frac{m!}{(m-k)!}$$

Per $n = 6$ e $m = 9$ (il caso più grande dell'esperimento) sono 207 775 stati. Per $n = 4$ e $m = 6$ sono 1 045.
Il limite `config.LIMITE_NODI = 500 000` è sopra il massimo, ma la ricerca termina prima in pratica (§5).

### 2.2 Costo minimo di un'assegnazione: $\varepsilon = 1$

La Proposizione 2.1 del corso chiede costi maggiori di un $\varepsilon > 0$, altrimenti la ricerca potrebbe non
terminare. Un incidente nella stessa cella della stazione darebbe $T = 0$. Per questo ogni assegnazione
costa almeno $\varepsilon = 1$:

$$C(m, i) = \max(T(m, i), 1)$$

Il costo della soluzione è quindi $\sum C(m,i)$ e non $\sum T(m,i)$. La differenza vale solo per gli
incidenti nella stessa cella della stazione, e tutti gli algoritmi usano lo stesso $C$, quindi il confronto
resta equo. Il costo minimo è riportato in `COSTO_MINIMO` e verificato in `test_costo_minimo_di_una_assegnazione_e_uno`.

### 2.3 Problema semplificato 1: si rimuove l'esclusività dei mezzi

Il problema reale chiede di assegnare gli incidenti residui $R = \{i \notin s\}$ a mezzi distinti.
Il rilassamento ignora l'esclusività: ogni incidente residuo prende il mezzo idoneo più vicino, anche se occupato
o già assegnato a un altro incidente.

$$h_{\min}(s) = \sum_{i \in R} \mu_i \qquad \mu_i = \min_{m \in \text{idonei}(i)} C(m, i)$$

- **Ammissibilità.** Ogni soluzione reale è anche una soluzione del rilassato, con lo stesso costo.
  Il rilassato ha più soluzioni, quindi il suo ottimo non supera quello reale.
- **Consistenza.** Il valore $\mu_i$ è statico: non dipende dallo stato. Un'azione che serve $i$ con il mezzo $m$
  riduce $h$ di esattamente $\mu_i \le C(m, i)$. Quindi $h(s) \le C(m,i) + h(s')$.

### 2.4 Problema semplificato 2: esclusività solo tra incidenti residui

Il secondo rilassamento è più stretto. Ammette che più incidenti residui usino lo stesso mezzo, ma i mezzi già
occupati dallo stato restano esclusi.

$$h_{\text{libero}}(s) = \sum_{i \in R} \min_{m \in \text{idonei}(i) \cap L(s)} C(m, i)$$

dove $L(s)$ è l'insieme dei mezzi liberi in $s$. Se per un incidente residuo non ci sono mezzi liberi idonei,
$h_{\text{libero}}(s) = \infty$: lo stato è un vicolo cieco e viene scartato.

- **Ammissibilità.** Nella soluzione reale ogni incidente residuo usa un mezzo libero idoneo, quindi costa almeno
  il minimo su $\text{idonei}(i) \cap L(s)$.
- **Consistenza.** Sia $i$ l'incidente servito dall'azione con il mezzo $m$, costo $c = C(m, i)$.
  Il mezzo $m$ esce dai liberi, quindi per ogni $j \in R \setminus \{i\}$ il minimo non diminuisce:

  $$h(s') \ge \sum_{j \in R\setminus\{i\}} \min_{L(s)} C(\cdot, j) = h(s) - \min_{L(s)} C(\cdot, i) \ge h(s) - c$$

  Quindi $h(s) \le c + h(s')$.

### 2.5 Confronto tra le due euristiche

Per ogni stato $h_{\text{libero}}(s) \ge h_{\min}(s)$: i minimi sono calcolati su un insieme più piccolo di mezzi.
Quindi $h_{\text{libero}}$ è più informativa, e A* con $h_{\text{libero}}$ espande un sottoinsieme di quello con $h_{\min}$.
Il test `test_dispatch_euristiche_ammissibili_consistenti_e_algoritmi_ottimi` verifica questa relazione su tutti
gli stati raggiungibili.

**Non si usa una euristica esatta.** Il sottoproblema residuo (assegnare gli incidenti ai mezzi liberi
con costo minimo) è un problema di assegnamento risolvibile con l'algoritmo ungherese. Risolverlo dentro $h$
significa risolvere il problema intero a ogni stato, e il confronto non avrebbe senso.
Per questo nessuna delle due euristiche è esatta: l'ottimo viene calcolato separatamente per la verifica.

### 2.6 Verifica esaustiva

Per istanze piccole il test calcola il costo ottimo esatto $V(s)$ di ogni stato raggiungibile con la
programmazione dinamica sugli stati, e controlla:

- $h(s) \le V(s)$ per ogni stato (ammissibilità);
- $h(s) \le C(m,i) + h(s')$ per ogni arco (consistenza);
- A* con $h_{\min}$, A* con $h_{\text{libero}}$ e LcFS trovano tutti $V(s_0)$ (ottimalità).

Sono 25 istanze casuali con 2–4 incidenti, 2–5 mezzi e costi in $[1, 6]$ con alcuni archi irraggiungibili.

---

## 3. Algoritmi

### 3.1 Non informato: LcFS

LcFS (Lowest Cost-First Search, detta anche Uniform-Cost Search, appunti §2.5.4) mantiene la frontiera come
coda a priorità ordinata per $g$ = costo del cammino. Con costi positivi è ottimale. È il riferimento non informato:
usa lo stesso ciclo di A* con $h = 0$, quindi il confronto isola l'effetto dell'euristica.

BFS minimizza il numero di archi, non il costo, quindi con costi $C$ non è ottimale. Non è usata nel confronto principale.

### 3.2 Informato: A*

A* ordina la frontiera per $f = g + h$. Con $h$ ammissibile e consistente:

- il primo obiettivo estratto è ottimale (appunti §2.6.3);
- la lista dei chiusi non riapre mai un nodo, quindi non serve riaprire i nodi (Proposizione 2.2);
- A* espande i nodi con $f < C^*$ e alcuni con $f = C^*$. Una euristica più alta, sempre ammissibile, riduce il primo insieme (appunti, "Importanza dell'euristica").

### 3.3 Alternative considerate

| Algoritmo | Ottimale | Memoria | Motivo della scelta |
|---|---|---|---|
| **A\*** (usato) | sì | $O(b^d)$ | ottimale, espande il minimo tra gli algoritmi ottimali |
| **LcFS** (usato) | sì | $O(b^d)$ | riferimento non informato, stesso ottimo |
| Greedy best-first | no | $O(b^d)$ | non ottimale: non confrontabile sulla qualità della soluzione |
| A\* pesato, $f = g + w\,h$ | no | $O(b^d)$ | accetta soluzioni subottimali: non ci interessa per il dispatch |
| Branch and bound | sì | $O(bd)$ | memoria lineare, da usare solo se `LIMITE_NODI` viene superato |
| IDA\* | sì | $O(bd)$ | rivisita i nodi, più costoso con costi reali |

Con $n \le 6$ lo spazio è piccolo rispetto alla memoria disponibile, quindi la memoria non è il vincolo principale
e A* resta la scelta. Branch and bound e IDA\* restano documentati come alternative per istanze più grandi.

---

## 4. Protocollo sperimentale

- Dimensioni: $n \in \{2, 3, 4, 5, 6\}$ incidenti (`config.N_INCIDENTI_SCENARI`).
- Mezzi: $\lceil 1.5 \cdot n \rceil$ (`config.N_MEZZI_PER_INCIDENTE`), di tipo uniforme tra i quattro della `TipoMezzo`.
- Scenari: 20 per dimensione (`config.N_SCENARI_PER_DIMENSIONE`), generati dal seme `config.SEME`.
- Griglia: $20 \times 20$, ostacoli con probabilità 0.2. Stazioni e incidenti in celle libere casuali.
- Uno scenario è valido se ogni incidente ha almeno un mezzo idoneo e raggiungibile. Gli scenari non validi
  vengono ricampionati.
- Per ogni scenario si eseguono LcFS, A* con $h_{\min}$ e A* con $h_{\text{libero}}$. Il test verifica che i costi
  siano uguali; se non lo sono, l'esperimento si ferma.
- Metriche: nodi espansi, nodi generati, tempo, numero di casi completati entro `LIMITE_NODI`.
- Output: `results/search_confronto.csv` (una riga per scenario e algoritmo) e `results/search_riepilogo.csv`
  (media ± deviazione standard per dimensione).

Esecuzione: `python src/search.py`.

---

## 5. Risultati

Esperimento eseguito con i parametri di `config.py`. Tutti i 100 casi per algoritmo sono completati
entro `LIMITE_NODI`, e i tre algoritmi trovano lo stesso costo ottimo su ogni scenario.

Nodi espansi, media ± deviazione standard su 20 scenari:

| $n$ | LcFS | A\* $h_{\min}$ | A\* $h_{\text{libero}}$ |
|---|---|---|---|
| 2 | 3.9 ± 0.8 | 3.5 ± 0.7 | 2.8 ± 1.1 |
| 3 | 17.4 ± 8.3 | 11.9 ± 5.3 | 7.0 ± 3.3 |
| 4 | 71.3 ± 45.2 | 33.7 ± 10.7 | 17.8 ± 8.4 |
| 5 | 373.4 ± 329.4 | 136.1 ± 93.7 | 62.4 ± 52.6 |
| 6 | 1884.0 ± 1977.7 | 856.2 ± 569.3 | 367.9 ± 321.0 |

**Lettura.** $h_{\text{libero}}$ riduce gli espansi di circa cinque volte rispetto a LcFS per $n = 6$, e di circa
2.3 volte rispetto a $h_{\min}$. Il vantaggio cresce con $n$, come previsto dalla teoria. La deviazione standard
è grande rispetto alla media: la difficoltà dipende molto dalla disposizione dei mezzi, e lo scenario conta più
dell'algoritmo. I tempi restano sotto i 0.02 secondi per caso.

---

## 6. Limiti

- $C = \max(T, 1)$ altera il costo quando un mezzo è nella stessa cella dell'incidente (§2.2).
- La griglia è uniforme, senza velocità diverse per tipo di strada: il costo è il numero di passi.
- Il rilassamento ignora la competizione tra incidenti per lo stesso mezzo, che è la parte difficile del problema.
  Per questo il vantaggio di A* è moderato per $n$ piccolo.
- Il numero di scenari (20 per dimensione) è quello di `config.py`. Con più scenari le deviazioni standard si stimano meglio.
- $h_{\min}$ e $h_{\text{libero}}$ sono consistenti sotto l'ipotesi che i costi $C(m,i)$ siano statici.
  Se in futuro i costi dipenderanno dallo stato (ad esempio dal traffico), le dimostrazioni di §2.3–2.4 vanno riviste.

---

## 7. Riferimenti al capitolo del corso

- Definizioni di spazio di stati, grafo e soluzione: §2.2–2.3.
- Algoritmo di ricerca generico e test dell'obiettivo alla selezione: §2.4.
- LcFS: §2.5.4. A*: §2.6.3. Euristica e problema semplificato: §2.6.1, §2.6.5.
- Ammissibilità e consistenza: §2.6.2–2.6.3. Proposizioni 2.1 (condizioni di A*) e 2.2 (consistenza e MPP).
- Potatura dei cammini multipli: §2.7.2. Confronto delle complessità: Tabella 2.1.
