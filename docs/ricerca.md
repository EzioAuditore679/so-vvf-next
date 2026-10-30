# Ricerca e assegnazione dei mezzi

Documento di riferimento per `src/ricerca.py`, `src/assegnazioni.py` e per i test `tests/test_ricerca.py` e
`tests/test_assegnazioni.py`. I rimandi (§) sono ai capitoli "Ricerca in Spazi di Stati" (§2) e
"Ragionamento con Vincoli" (§3) delle dispense del corso.

Il dispatch si divide in due problemi, ciascuno in un proprio modulo:

| Modulo | Problema | Tecnica | Cosa si confronta |
|---|---|---|---|
| `ricerca.py` | percorso del mezzo dalla stazione all'incidente | ricerca su grafo | BFS, LcFS, A* con due euristiche, una costruita con la programmazione dinamica |
| `assegnazioni.py` | quale mezzo copre quale ruolo della squadra | ottimizzazione vincolata con simulated annealing | distanza dall'ottimo esatto, con un tentativo e con ripartenze |

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

## 2. Assegnazione dei mezzi: ottimizzazione vincolata

### 2.1 Formulazione

| Elemento | Definizione |
|---|---|
| Variabili | i ruoli richiesti dagli incidenti (estinzione, rinforzo, soccorso in quota, decontaminazione) |
| Domini | i mezzi adatti al ruolo secondo la KB e raggiungibili, più il valore *scoperto* |
| Vincolo rigido | un mezzo copre al più un ruolo |
| Funzione obiettivo | somma dei tempi di percorrenza, più $P$ per ogni ruolo scoperto |

È un problema di ottimizzazione vincolato (§3.10). *Scoperto* significa chiedere un rinforzo a un comando vicino,
al costo $P$ = `config.PENALITA_RUOLO_SCOPERTO` = 120. Il valore supera il tempo di percorrenza più lungo osservato
sulla mappa 30×30 (86), quindi un ruolo resta scoperto solo se non c'è un mezzo adatto libero.

Come suggerito in §3.10, il vincolo rigido diventa un costo alto ma finito: ogni volta che un mezzo è usato
più di una volta si aggiunge $C$ = `config.COSTO_CONFLITTO` = $P + 1$. Con $C > P$ un conflitto si risolve sempre
lasciando scoperto uno dei due ruoli, con un guadagno; con $C$ finito e vicino a $P$ la ricerca può attraversare
assegnazioni con un conflitto, ad esempio per scambiare due mezzi tra due ruoli. Si restituisce comunque la
migliore assegnazione **senza** conflitti visitata. Un primo tentativo con $C$ maggiore della somma di tutti gli
altri costi (la variante rigida di §3.10) rendeva gli scambi quasi impossibili a bassa temperatura e lasciava
soluzioni peggiori.

### 2.2 Simulated annealing con ripartenze casuali

Si usa la ricerca locale per l'ottimizzazione (§3.10.2) nella forma del simulated annealing (§3.7):

- **assegnazione iniziale casuale**, un valore del dominio per ogni ruolo;
- **successore**: si cambia il valore di un solo ruolo, scelti a caso il ruolo e il nuovo valore;
- **accettazione**: il passo si accetta se non peggiora il costo, altrimenti con probabilità $e^{-(h(A') - h(A))/T}$;
- **raffreddamento geometrico** da $T_0$ a $T_f$ in un numero fisso di passi;
- **ripartenze casuali** (§3.7): più tentativi indipendenti, si tiene il migliore.

Poiché un passo cambia un solo ruolo, la variazione di costo si calcola guardando solo i due mezzi coinvolti
(quello lasciato e quello preso), senza ricalcolare tutta l'assegnazione.

La ricerca locale è incompleta: non garantisce l'ottimo. Per misurare quanto se ne allontana si usa come
riferimento l'ottimo esatto: con questi vincoli il problema è un assegnamento, risolvibile in tempo polinomiale
con l'algoritmo ungherese (`ottimo_esatto`, tramite scipy). L'ottimo esatto serve solo alla valutazione e ai test;
la formulazione come ottimizzazione vincolata resta utile perché accetta vincoli che rompono la struttura di
assegnamento, ad esempio mezzi incompatibili sullo stesso incidente o un tempo massimo per i ruoli critici.

### 2.3 Scelta dei parametri

Taratura su 32 scenari (4, 6, 8 e 10 incidenti) generati con un seme diverso da quello della valutazione
(`config.SEME + 500`), un solo tentativo per scenario. Percentuale di soluzioni ottime e scarto medio dall'ottimo:

| Passi | $T_0 = 30$ | $T_0 = 100$ | $T_0 = 300$ |
|---|---|---|---|
| 20 000 | 28% / 5.5% | 31% / 3.7% | 38% / 4.5% |
| 50 000 | 44% / 2.8% | 53% / 1.6% | 47% / 2.6% |
| 100 000 | 53% / 1.8% | 63% / 1.0% | 50% / 1.9% |

- $T_0 = 100$ è il migliore per ogni numero di passi: a inizio ricerca un peggioramento di 10 minuti si accetta con
  probabilità 0.9, uno pari alla penalità (120) con probabilità 0.3. $T_f = 0.1$: alla fine non si accettano quasi più peggioramenti.
- **50 000 passi** per tentativo (circa 0.06 s): raddoppiarli migliora poco lo scarto (dall'1.6% all'1.0%) e
  raddoppia il tempo; conviene spendere il tempo in **5 ripartenze**, che misurano anche quanto aiuta la ripartenza casuale.

### 2.4 Verifica

`tests/test_assegnazioni.py`: l'ottimo esatto coincide con l'enumerazione di tutte le assegnazioni su 20 istanze
piccole; su 30 istanze casuali il simulated annealing restituisce sempre un'assegnazione ammissibile (mezzi distinti
e adatti, ogni ruolo deciso una volta) e trova l'ottimo; un conflitto costa più di un ruolo scoperto.

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
| Simulated annealing | $T_0 = 100$, $T_f = 0.1$, 50 000 passi, 5 ripartenze | taratura in §2.3 |

Esecuzione: `python src/ricerca.py` e `python src/assegnazioni.py`. Risultati in `results/ricerca_*.csv` e
`results/assegnazioni_*.csv`, grafico `plots/ricerca_percorsi.png`.

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

Simulated annealing confrontato con l'ottimo esatto, 20 scenari per dimensione (seme di valutazione):

| Incidenti | Ruoli | 1 tentativo: ottimi | 1 tentativo: scarto | 5 tentativi: ottimi | 5 tentativi: scarto |
|---|---|---|---|---|---|
| 2 | 2.8 | 100% | 0 | 100% | 0 |
| 4 | 6.2 | 100% | 0 | 100% | 0 |
| 6 | 9.8 | 80% | 0.9% ± 2.2% | 100% | 0 |
| 8 | 12.8 | 30% | 4.3% ± 6.4% | 55% | 1.3% ± 2.2% |
| 10 | 16.2 | 15% | 3.5% ± 3.1% | 40% | 1.3% ± 1.6% |

**Lettura.**
- Fino a 6 incidenti contemporanei le 5 ripartenze trovano sempre l'ottimo; la valutazione end-to-end usa 5 incidenti.
- Con 8–10 incidenti la soluzione resta in media l'1.3% sopra l'ottimo, cioè pochi minuti su tempi totali di
  qualche centinaio. Le ripartenze dimezzano o più lo scarto di un singolo tentativo, come previsto da
  $1 - (1 - p)^n$ (§3.7): se un tentativo trova l'ottimo con probabilità $p$, cinque lo trovano più spesso.

---

## 4. Limiti

- La mappa è una griglia con quattro mosse e due soli costi per cella; le strade vere hanno velocità diverse e sensi unici.
- La tabella della PD ignora le chiusure: con molte chiusure il suo vantaggio si riduce (§3.2).
- Il simulated annealing non garantisce l'ottimo: con molti incidenti contemporanei resta qualche punto percentuale
  sopra (§3.3). Senza vincoli aggiuntivi l'assegnamento si risolverebbe esattamente in tempo polinomiale (§2.2).
- I tempi di percorrenza sono statici: un mezzo assegnato non cambia la mappa per gli altri.
