# Ricerca nello spazio degli stati

Documento di riferimento per `src/search.py` e `tests/test_search.py`. Segue il capitolo
"Ricerca in Spazi di Stati" del corso (sezioni 2.2–2.8, Proposizioni 2.1 e 2.2, Tabella 2.1).

Il problema è risolto su due livelli. Il livello 1 calcola le distanze tra stazioni e incidenti.
Il livello 2 decide quali mezzi coprono i ruoli richiesti da ciascun incidente, usando quelle distanze come costi.

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

## 2. Livello 2: copertura dei ruoli della squadra

### 2.1 Rappresentazione

La gravità di un incidente decide la squadra. La KB Prolog (`ruolo/2` in `knowledge_base/kb.pl`) deriva
i ruoli richiesti: `estinzione` sempre, `rinforzo` se la gravità operativa è critica, `soccorso_quota`
con persone ai piani alti (le persone incerte contano solo se la gravità non è bassa),
`decontaminazione` con sostanze pericolose. La KB dà anche i mezzi adatti a ogni ruolo (`copre/3`).
Sul modello generativo un incidente richiede in media 1.6 ruoli.

Il dispatch è un CSP risolto con la ricerca: una variabile per ruolo, i valori sono i mezzi adatti
e il valore speciale *scoperto*.

| Elemento | Definizione |
|---|---|
| Variabili | ruoli $r_1, \ldots, r_R$, coppie (incidente, ruolo), in ordine fisso (§2.2) |
| Stati | $s = (k, O)$: i primi $k$ ruoli sono decisi, $O$ è l'insieme dei mezzi occupati |
| Stato iniziale | $(0, \emptyset)$ |
| Azioni sul ruolo $r_{k+1}$ | un mezzo $m \notin O$ adatto e raggiungibile, costo $C(m, i)$; oppure *scoperto*, costo $P$ |
| Obiettivo | $k = R$ |
| Costo di una soluzione | $\sum C(m, i)$ sulle assegnazioni $+\; P \cdot$ (ruoli scoperti) |

**Lo stato non memorizza le assegnazioni.** Due assegnazioni parziali con lo stesso $k$ e gli stessi
mezzi occupati hanno lo stesso futuro: stessi ruoli residui, stessi mezzi liberi. Sono quindi lo stesso
stato, e la lista dei chiusi elimina i cammini multipli (§2.7.2 del corso). Le assegnazioni si ricostruiscono
dal cammino (`Dispatch.piano`).

**Ruolo scoperto.** Se la flotta non basta, un ruolo resta scoperto e si chiede un rinforzo a un comando
vicino, al costo fisso $P$ = `config.PENALITA_RUOLO_SCOPERTO` = 60. Il valore supera il tempo di
percorrenza tipico sulla griglia $20 \times 20$, quindi la ricerca lascia scoperto un ruolo solo quando
non ha alternative ragionevoli. Ogni scenario ha così una soluzione, e la valutazione end-to-end (§6)
può contare i ruoli scoperti invece di scartare gli scenari.

### 2.2 Ordine delle variabili: MRV statico

I ruoli sono ordinati una volta per tutte per numero crescente di mezzi ammessi (minimum remaining values,
dal capitolo sui CSP). L'ordine non cambia l'ottimo, perché ogni assegnazione completa resta raggiungibile
esattamente una volta. Riduce i rami: i ruoli con un solo mezzo possibile, ad esempio il soccorso in quota
con una sola autoscala, vengono decisi per primi.

### 2.3 $\varepsilon = 1$

La Proposizione 2.1 chiede costi maggiori di un $\varepsilon > 0$. Un mezzo nella stessa cella
dell'incidente avrebbe $T = 0$, quindi $C(m, i) = \max(T(m, i), 1)$. Anche $P \ge 1$.
Verificato in `test_costo_minimo_di_una_assegnazione_e_uno`.

### 2.4 Problema semplificato 1: si rimuove l'esclusività dei mezzi

Ogni ruolo residuo prende il valore più economico tra i mezzi adatti, anche se già occupati, e la penalità:

$$h_{\min}(k, O) = \sum_{j > k} \mu_j \qquad \mu_j = \min\Big(P,\ \min_{m \in \text{adatti}(r_j)} C(m, i_j)\Big)$$

- **Ammissibilità.** Ogni completamento reale è anche un completamento del problema rilassato, con lo
  stesso costo; il rilassato ha più soluzioni, quindi il suo ottimo non supera quello reale.
- **Consistenza.** $\mu_j$ non dipende dallo stato. L'azione sul ruolo $r_{k+1}$ costa almeno $\mu_{k+1}$
  e riduce $h$ di esattamente $\mu_{k+1}$, quindi $h(s) \le c + h(s')$.

### 2.5 Problema semplificato 2: esclusività solo verso i mezzi già occupati

I mezzi in $O$ restano esclusi; tra i ruoli residui un mezzo libero può essere usato più volte:

$$h_{\text{libero}}(k, O) = \sum_{j > k} \min\Big(P,\ \min_{m \in \text{adatti}(r_j) \setminus O} C(m, i_j)\Big)$$

- **Ammissibilità.** Nel completamento reale ogni ruolo residuo usa un mezzo libero adatto oppure la penalità,
  quindi costa almeno il minimo calcolato su $\text{adatti}(r_j) \setminus O$ e $P$.
- **Consistenza.** L'azione sul ruolo $r_{k+1}$ ha costo $c$. Il termine di $r_{k+1}$ esce dalla somma ed è
  $\le c$, perché $c$ è uno dei valori su cui si prende il minimo. Gli altri termini non diminuiscono, perché
  $O$ può solo crescere. Quindi $h(s') \ge h(s) - c$.

Per ogni stato $h_{\text{libero}} \ge h_{\min}$, quindi A* con $h_{\text{libero}}$ è più informato.
Grazie alla penalità nessuna delle due vale $\infty$.

**Non si usa un'euristica esatta.** Il sottoproblema residuo è un problema di assegnamento risolvibile con
l'algoritmo ungherese: risolverlo dentro $h$ significa risolvere il problema intero a ogni stato.

### 2.6 Verifica esaustiva

`test_dispatch_euristiche_ammissibili_consistenti_e_algoritmi_ottimi`, su 30 istanze casuali con 2–4 incidenti,
1–3 ruoli ciascuno, 2–6 mezzi e alcuni archi irraggiungibili, calcola il costo ottimo $V(s)$ di ogni stato
raggiungibile con la programmazione dinamica e controlla:

- $h(s) \le V(s)$ per ogni stato, e $h_{\min}(s) \le h_{\text{libero}}(s)$;
- $h(s) \le c + h(s')$ per ogni arco;
- LcFS, A* con $h_{\min}$ e A* con $h_{\text{libero}}$ trovano $V(s_0)$, e il piano ricostruito usa mezzi
  distinti e ha esattamente quel costo.

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

Con $n \le 10$ lo spazio resta sotto `LIMITE_NODI` (il caso peggiore, LcFS a $n = 10$, espande circa 472 000 nodi: vicino al limite, quindi $n = 10$ è la dimensione massima dell'esperimento), quindi la memoria non è il vincolo principale
e A* resta la scelta. Branch and bound e IDA\* restano documentati come alternative per istanze più grandi.

---

## 4. Protocollo sperimentale

- Dimensioni: $n \in \{2, 4, 6, 8, 10\}$ incidenti (`config.N_INCIDENTI_SCENARI`), 20 scenari ciascuna,
  generati dal seme `config.SEME`.
- Incidenti dal modello generativo; ruoli e mezzi adatti dalla KB, con la velocità vera del fuoco.
  Due incidenti a distanza di Manhattan $\le 3$ sono adiacenti, quindi la propagazione del pericolo
  (`esposto/1`) può aggiungere un rinforzo.
- Flotta: $\lceil 2.2\, n \rceil$ mezzi, con tipo estratto dalla composizione `config.COMPOSIZIONE_FLOTTA`
  (autopompe 50%, NBCR 20%, canadair 15%, autoscale 15%), proporzionale alla domanda media dei ruoli.
- Griglia $20 \times 20$ con ostacoli al 20%. Costi dal livello 1.
- Su ogni scenario LcFS, A* $h_{\min}$ e A* $h_{\text{libero}}$; l'esperimento si ferma se due algoritmi
  completi danno costi diversi.
- Output: `results/search_confronto.csv` e `results/search_riepilogo.csv`. Esecuzione: `python src/search.py`.

---

## 5. Risultati

Tutti i 100 casi per algoritmo sono completati entro `LIMITE_NODI`, con lo stesso costo ottimo.
Nodi espansi, media ± deviazione standard su 20 scenari:

| $n$ | ruoli (media) | LcFS | A\* $h_{\min}$ | A\* $h_{\text{libero}}$ |
|---|---|---|---|---|
| 2 | 2.9 | 4.4 ± 1.6 | 3.4 ± 1.3 | 3.0 ± 1.2 |
| 4 | 6.8 | 48.5 ± 52.0 | 25.0 ± 29.1 | 19.0 ± 24.0 |
| 6 | 10.8 | 922 ± 1843 | 550 ± 1233 | 432 ± 1006 |
| 8 | 13.7 | 3204 ± 4467 | 1460 ± 2868 | 1142 ± 2733 |
| 10 | 16.6 | 78 800 ± 121 571 | 45 979 ± 85 564 | 32 047 ± 65 228 |

**Lettura.**
- A* con $h_{\text{libero}}$ espande circa 2.5 volte meno nodi di LcFS a $n = 10$, e il rapporto cresce
  con $n$. $h_{\min}$ sta in mezzo, come previsto da $h_{\min} \le h_{\text{libero}}$.
- Il guadagno in tempo è minore di quello in nodi: $h_{\text{libero}}$ costa $O(R \cdot m)$ per nodo,
  $h_{\min}$ è una lettura di tabella. A $n = 10$: 2.3 s per caso con $h_{\text{libero}}$,
  1.4 s con $h_{\min}$, 2.7 s con LcFS.
- La deviazione standard supera la media: la difficoltà dipende soprattutto da quanti ruoli competono
  per gli stessi mezzi scarsi (autoscale, NBCR), cioè dallo scenario più che dall'algoritmo.
- In media restano scoperti tra 0.6 e 1.8 ruoli per scenario: la flotta casuale non sempre ha il tipo di mezzo richiesto.
- Rispetto alla formulazione precedente (un mezzo per incidente, stato = insieme di assegnazioni) lo spazio è
  molto più piccolo a parità di $n$, grazie allo stato compresso e all'ordine MRV.

---

## 6. Limiti

- $C = \max(T, 1)$ altera il costo quando un mezzo è nella stessa cella dell'incidente (§2.3).
- La penalità $P$ è un parametro: con $P$ più basso la ricerca lascerebbe scoperti ruoli costosi da raggiungere.
- La griglia è uniforme, senza velocità diverse per tipo di strada.
- Il rilassamento ignora la competizione tra ruoli residui per lo stesso mezzo, che è la parte difficile.
- Le euristiche sono consistenti sotto l'ipotesi di costi statici.

---

## 7. Riferimenti al capitolo del corso

- Definizioni di spazio di stati, grafo e soluzione: §2.2–2.3.
- Algoritmo di ricerca generico e test dell'obiettivo alla selezione: §2.4.
- LcFS: §2.5.4. A*: §2.6.3. Euristica e problema semplificato: §2.6.1, §2.6.5.
- Ammissibilità e consistenza: §2.6.2–2.6.3. Proposizioni 2.1 (condizioni di A*) e 2.2 (consistenza e MPP).
- Potatura dei cammini multipli: §2.7.2. Confronto delle complessità: Tabella 2.1.
- Ordine delle variabili (MRV) e ricerca su assegnazioni parziali: capitolo sui CSP.
