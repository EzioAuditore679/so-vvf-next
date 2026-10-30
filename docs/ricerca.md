# Ricerca e assegnazione dei mezzi

Documento di riferimento per `src/ricerca.py`, `src/dispatch.py` e per i test `tests/test_ricerca.py` e
`tests/test_dispatch.py`. I rimandi (§) sono al capitolo "Ricerca in Spazi di Stati" delle dispense del corso.

Il dispatch si divide in due problemi, ciascuno in un proprio modulo:

| Modulo | Problema | Tecnica | Cosa si confronta |
|---|---|---|---|
| `ricerca.py` | percorso del mezzo dalla stazione all'incidente | ricerca su grafo | BFS, LcFS, A* con due euristiche, una costruita con la programmazione dinamica |
| `dispatch.py` | quale mezzo copre quale ruolo della squadra | CSP risolto con branch and bound | branch and bound con e senza stima |

I ruoli della squadra e i mezzi adatti a ciascun ruolo vengono dalla KB Prolog (`ruolo/2`, `copre/3`).
I tempi di percorrenza che l'assegnazione minimizza vengono da `ricerca.py`.

---

## 1. Percorso del mezzo

### 1.1 Rappresentazione

| Elemento | Definizione |
|---|---|
| Stati | celle libere della mappa 30×30 |
| Stato iniziale / obiettivo | cella della stazione del mezzo / cella dell'incidente |
| Azioni | un passo nelle 4 direzioni verso una cella libera |
| Costo di un passo | costo della cella di arrivo: 1, oppure 3 per una cella lenta |

La mappa ha tre tipi di celle speciali, con significati diversi:

- **ostacoli permanenti** (20%): edifici, noti in anticipo;
- **celle lente** (15% delle libere, costo 3): centro storico, traffico; note in anticipo;
- **chiusure temporanee**: strade chiuse il giorno della chiamata, note solo quando arriva la chiamata.

Le celle lente rendono diversi il percorso con meno passi e quello più veloce: per questo nel confronto c'è
anche la BFS. Le chiusure separano ciò che si sa prima della chiamata da ciò che si sa solo dopo: è la
distinzione su cui si costruisce l'euristica con la programmazione dinamica (§1.3).

### 1.2 Algoritmi

Tutti usano la potatura dei cammini multipli con lista dei chiusi (§2.7.2).

- **BFS**: frontiera FIFO; trova il cammino con meno passi, che con le celle lente non è il più veloce.
- **LcFS**: frontiera ordinata per costo; ottimale, nessuna informazione sull'obiettivo. È il riferimento.
- **A\***: frontiera ordinata per costo + h. Stesso codice di LcFS (`a_star` con h = 0 è LcFS), quindi
  il confronto isola l'effetto dell'euristica.

### 1.3 Euristiche da problemi semplificati

Entrambe nascono dalla procedura di §2.6.5: si toglie un vincolo e si usa il costo ottimo del problema più semplice.

| Euristica | Problema semplificato | Come si calcola |
|---|---|---|
| $h_M$ (Manhattan) | niente ostacoli, niente chiusure, ogni cella costa 1 | in forma chiusa: $\lvert r - r_g\rvert + \lvert c - c_g\rvert$ |
| $h_{PD}$ | la mappa vera **senza le chiusure temporanee** | programmazione dinamica (§2.8.4) |

**Calcolo di $h_{PD}$.** Per l'incidente $g$ si calcola una sola volta la tabella

$$\text{cost\_to\_goal}(n) = \begin{cases} 0 & n = g \\ \min_{\langle n, m\rangle} \big[\,\text{cost}(n, m) + \text{cost\_to\_goal}(m)\,\big] & \text{altrimenti} \end{cases}$$

con LcFS sul grafo inverso a partire da $g$, sulla mappa senza chiusure (`cost_to_goal` in `ricerca.py`).
La tabella è il DB di pattern dell'incidente; A* la legge in tempo costante. Una cella irraggiungibile anche
senza chiusure lo è anche sulla mappa vera: per lei $h_{PD} = \infty$ e A* non la espande.

**Ammissibilità e consistenza, in una sola proprietà.** Sia $h$ il costo esatto verso $g$ in un grafo che
contiene tutti gli archi della mappa vera, con costi non maggiori.
- *Ammissibile*: ogni cammino della mappa vera esiste anche nel grafo semplificato e non costa di più, quindi
  $h(n) \le h^*(n)$.
- *Consistente*: per ogni arco vero $\langle n, n'\rangle$, l'arco esiste anche nel grafo semplificato, e un costo
  esatto rispetta la disuguaglianza triangolare: $h(n) \le \text{cost}(n, n') + h(n')$.

Entrambe le euristiche rientrano in questo caso: la mappa senza chiusure ha più archi della mappa vera
($h_{PD}$); la griglia vuota a costo 1 ne ha ancora di più e costi più bassi ($h_M$). Poiché il problema di
$h_M$ semplifica quello di $h_{PD}$, vale anche $h_M \le h_{PD} \le h^*$: $h_{PD}$ è più informata.
Senza chiusure $h_{PD} = h^*$: è l'euristica perfetta del corso, e A* espande solo celle su cammini ottimi.

Con un'euristica consistente la potatura dei cammini multipli non perde l'ottimo (Proposizione 2.2).

**Costo della programmazione dinamica.** La tabella espande tutte le celle raggiungibili (circa 715 per
incidente sulla mappa 30×30) e si calcola una volta per incidente, poi la usano tutti i mezzi diretti lì.
Conviene quando il risparmio sui mezzi supera questo costo (§3.2). In alternativa, poiché ostacoli e celle
lente sono noti prima delle chiamate, si possono precalcolare offline le tabelle di tutte le celle
(circa 720 tabelle da 720 valori): online il costo diventa nullo, come nell'uso offline descritto in §2.8.4.

### 1.4 Verifica

`tests/test_ricerca.py`, su 20 mappe casuali con ostacoli, celle lente e chiusure: per ogni cella
$h_M \le h_{PD} \le h^*$, con $h^*$ calcolato con la stessa PD sulla mappa vera; la consistenza su ogni arco;
lo stesso costo ottimo per LcFS, A* $h_M$ e A* $h_{PD}$; la BFS mai migliore dell'ottimo, e uguale all'ottimo
quando le celle costano tutte 1.

---

## 2. Assegnazione dei mezzi come CSP

### 2.1 Formulazione

| Elemento | Definizione |
|---|---|
| Variabili | i ruoli richiesti dagli incidenti (estinzione, rinforzo, soccorso in quota, decontaminazione) |
| Domini | i mezzi adatti al ruolo secondo la KB e raggiungibili, più il valore *scoperto* |
| Vincolo | un mezzo copre al più un ruolo |
| Obiettivo | minimizzare la somma dei tempi di percorrenza, più $P$ per ogni ruolo scoperto |

*Scoperto* significa chiedere un rinforzo a un comando vicino, al costo $P$ = `config.PENALITA_RUOLO_SCOPERTO` = 120.
Il valore supera il tempo di percorrenza più lungo osservato sulla mappa 30×30 (86), quindi un ruolo resta
scoperto solo se non c'è un mezzo adatto libero. Grazie a questo valore ogni scenario ha una soluzione.

### 2.2 Branch and bound in profondità

Si usa il branch and bound di §2.6.4 sulle assegnazioni parziali, con tre tecniche dei CSP:

- **ordine delle variabili MRV**: si decide prima il ruolo con meno valori ancora ammessi; a parità, quello con il
  rimpianto più alto (differenza tra secondo e primo valore), cioè quello per cui sbagliare costa di più;
- **ordine dei valori**: dal mezzo più vicino, così la prima soluzione completa è già buona e alza subito il limite;
- **domini aggiornati** a ogni assegnazione (forward checking): i mezzi usati escono dai domini degli altri ruoli.
  Nessun dominio si svuota mai, perché *scoperto* resta sempre ammesso: l'aggiornamento serve a MRV e alla stima.

Un ramo si pota quando costo già speso + stima ≥ costo della migliore soluzione trovata. Il riferimento non
informato è lo stesso algoritmo con stima nulla, che pota solo con il costo già speso: è il ruolo che LcFS ha per A*.

### 2.3 La stima: rilassamenti del vincolo di esclusività

Il branch and bound richiede solo che la stima non superi il costo minimo dei ruoli residui. Si parte da
**h_libero**: ogni ruolo residuo prende il suo valore più economico tra mezzi liberi e penalità, anche se un altro
ruolo vuole lo stesso mezzo (si toglie l'esclusività tra i ruoli residui). A questa base si aggiungono due
correzioni, ciascuna ammissibile, applicate a gruppi di ruoli disgiunti:

- **capacità**: se $k$ ruoli possono usare solo un gruppo di $d$ mezzi e $k > d$, in ogni soluzione almeno $k - d$
  di loro restano scoperti e pagano $P$ invece del loro minimo. Si aggiungono gli aumenti più piccoli: è il minimo
  possibile.
- **mezzi contesi**: se più ruoli hanno come prima scelta lo stesso mezzo, uno solo può averlo; gli altri pagano almeno
  il loro secondo valore. Si lascia il mezzo al ruolo per cui perderlo costerebbe di più, quindi l'aumento è il minimo possibile.

Le due correzioni riguardano ruoli diversi, quindi le loro somme restano un limite inferiore.
`test_stima_ammissibile_in_ogni_stato` lo verifica in 600 stati casuali contro l'ottimo esatto.

**Nota sulla complessità.** Con questi vincoli l'assegnazione è un problema di assegnamento, risolvibile in tempo
polinomiale con l'algoritmo ungherese. I test lo usano per verificare che il branch and bound trovi l'ottimo
(`test_branch_and_bound_trova_l_ottimo`, 40 istanze). La formulazione CSP resta utile perché accetta vincoli che
rompono la struttura di assegnamento, ad esempio mezzi incompatibili tra loro sullo stesso incidente o un tempo
massimo di arrivo per i ruoli critici.

---

## 3. Protocollo e risultati

### 3.1 Parametri (`config.py`)

| Parametro | Valore | Motivazione |
|---|---|---|
| Mappa | 30×30, ostacoli 20% | abbastanza grande da separare gli algoritmi, la PD resta sotto i 1000 nodi |
| Celle lente | 15% delle libere, costo 3 | rendono la BFS non ottimale nel 54–72% dei casi |
| Chiusure | 0%, 5%, 10%, 20% | da nessuna informazione nuova a molta: misura quanto la PD perde precisione |
| Mappe | 30 per livello, 8 mezzi per mappa | 240 percorsi per livello; la tabella della PD è condivisa dagli 8 mezzi |
| Assegnazione | 2–10 incidenti, 20 scenari ciascuno, 2.2 mezzi per incidente | un incidente richiede in media 1.6 ruoli |
| Limite di nodi | 500 000 | oltre, il caso è registrato come non chiuso |

Esecuzione: `python src/ricerca.py` e `python src/dispatch.py`. Risultati in `results/ricerca_*.csv` e
`results/dispatch_*.csv`, grafici `plots/ricerca_percorsi.png` e `plots/assegnazione_bb.png`.

### 3.2 Percorso del mezzo

Nodi espansi per percorso, media ± dev. std. (240 percorsi per livello; a 20% di chiusure 146 sono raggiungibili).
LcFS e A* trovano sempre il percorso ottimo.

| Chiusure | BFS | LcFS | A* $h_M$ | A* $h_{PD}$ | BFS: tempo in più | BFS ottima |
|---|---|---|---|---|---|---|
| 0% | 343 ± 208 | 335 ± 206 | 92 ± 81 | 36 ± 26 | 14.4% | 28% |
| 5% | 338 ± 199 | 343 ± 203 | 112 ± 89 | 58 ± 50 | 11.3% | 35% |
| 10% | 331 ± 187 | 333 ± 187 | 130 ± 103 | 76 ± 73 | 10.2% | 29% |
| 20% | 227 ± 140 | 225 ± 138 | 106 ± 77 | 82 ± 64 | 6.0% | 46% |

**Lettura.**
- A* con $h_{PD}$ espande circa 9 volte meno nodi di LcFS senza chiusure, e circa 6 volte meno al 5%.
  Più chiusure ci sono, più la tabella (calcolata senza chiusure) si allontana dal vero e il vantaggio su
  Manhattan si riduce: da 2.5 volte a 1.3 volte al 20%.
- La BFS espande quanto LcFS ma sbaglia il tempo nel 54–72% dei casi, con il 6–14% in più in media: per il
  dispatch conta il tempo, quindi serve un algoritmo a costo minimo.
- **Ammortamento della PD.** La tabella costa circa 715 nodi per incidente. Rispetto a LcFS si ripaga dal terzo
  mezzo (risparmio di circa 300 nodi a mezzo); rispetto a Manhattan servono circa 13 mezzi per lo stesso
  incidente (risparmio di circa 55 nodi a mezzo). Calcolata online per ogni incidente, quindi, conviene solo
  contro la ricerca non informata; precalcolata offline (§1.3) conviene sempre.

### 3.3 Assegnazione dei mezzi

Nodi visitati dal branch and bound, media ± dev. std. su 20 scenari (casi chiusi entro 500 000 nodi):

| Incidenti | Ruoli | Senza stima | Con stima |
|---|---|---|---|
| 2 | 2.8 | 16 ± 12 (20/20) | 9 ± 2 (20/20) |
| 4 | 6.2 | 700 ± 1 623 (20/20) | 40 ± 37 (20/20) |
| 6 | 9.8 | 102 227 ± 189 734 (17/20) | 1 237 ± 2 623 (20/20) |
| 8 | 12.8 | 221 857 ± 185 172 (15/20) | 28 037 ± 111 250 (19/20) |
| 10 | 16.2 | 445 713 ± 119 772 (4/20) | 39 964 ± 112 850 (19/20) |

**Lettura.** La stima riduce i nodi di uno o due ordini di grandezza e chiude quasi tutti i casi. Con 8–10
incidenti contemporanei resta un caso su 20 che supera il limite: sono scenari in cui molti ruoli si contendono
mezzi a tempi quasi uguali, dove nessuna delle due correzioni della stima aiuta. La deviazione standard maggiore
della media indica che la difficoltà dipende dallo scenario più che dalla dimensione.
La valutazione end-to-end usa 5 incidenti per scenario, una dimensione in cui tutti i casi si chiudono.

---

## 4. Limiti

- La mappa è una griglia con quattro mosse e due soli costi per cella; le strade vere hanno velocità diverse e sensi unici.
- La tabella della PD ignora le chiusure: con molte chiusure il suo vantaggio si riduce (§3.2).
- Il branch and bound ha costo esponenziale nel caso peggiore; per istanze molto grandi l'assegnamento puro si risolve
  in tempo polinomiale (§2.3), a meno di vincoli aggiuntivi.
- I tempi di percorrenza sono statici: un mezzo assegnato non cambia la mappa per gli altri.
