%% knowledge_base/kb.pl
%% Knowledge base Prolog di SO-VVF-NEXT (interrogata da src/prolog_engine.py via pyswip).
%%
%% Vocabolario: gli stessi valori di src/domain.py (atomi minuscoli).
%% Le REGOLE sono fisse in questo file. I FATTI dinamici (incidenti, mezzi, registro
%% territoriale, velocita stimata, assegnazioni) vengono caricati per ogni scenario.
%%
%% Sezioni:
%%   1. Idoneita dei mezzi, con negazione per fallimento (\+)
%%   2. Gravita: stessa regola di modello_generativo.regola_gravita
%%   3. Abduzione: quali velocita del fuoco spiegano una gravita osservata
%%   4. Propagazione del pericolo: ricorsione sugli edifici adiacenti
%%   5. Vincoli di integrita sulle assegnazioni
%%   6. Statistiche della KB (per la relazione)

:- dynamic incidente/6.        % incidente(Id, Tipo, Piano, Sostanze, Persone, Meteo)
:- dynamic registro/3.         % registro(Id, Vulnerabilita, NAdiacenti)   [fatto del registro territoriale]
:- dynamic adiacenza/2.        % adiacenza(IdA, IdB), caricata in entrambi i versi
:- dynamic mezzo/2.            % mezzo(Id, TipoMezzo)
:- dynamic velocita_stimata/2. % velocita_stimata(Id, Velocita): ground truth o stima della rete bayesiana
:- dynamic assegnato/2.        % assegnato(IdMezzo, IdIncidente): proposta di dispatch da verificare

%% ---------------------------------------------------------------------------
%% 1. Idoneita dei mezzi
%% ---------------------------------------------------------------------------

% Soglia di altezza per l'autoscala. Uguale a domain.SOGLIA_PIANI_AUTOSCALA (verificato nei test).
soglia_piani_autoscala(4).

% requisito(TipoMezzo, TipoEdificio, Piano, Sostanze)
requisito(autopompa, _, _, _).                                  % sempre idonea
requisito(autoscala, Tipo, Piano, _) :-                         % serve in altezza, non nel bosco
    Tipo \== bosco,
    soglia_piani_autoscala(S),
    Piano >= S.
requisito(nucleo_nbcr, _, _, si).                               % solo con sostanze pericolose
requisito(canadair, bosco, _, _).                               % solo negli incendi boschivi

% Negazione per fallimento: il canadair non interviene dove ci sono sostanze pericolose
% (acqua su prodotti chimici). Incompatibilita esplicita.
incompatibile(M, I) :-
    mezzo(M, canadair),
    incidente(I, _, _, si, _, _).

% idoneo(M, I): il mezzo M puo essere assegnato all'incidente I.
idoneo(M, I) :-
    mezzo(M, T),
    incidente(I, Tipo, Piano, Sost, _, _),
    requisito(T, Tipo, Piano, Sost),
    \+ incompatibile(M, I).

mezzi_idonei(I, Lista) :-
    findall(M, idoneo(M, I), Lista).

%% ---------------------------------------------------------------------------
%% 2. Gravita
%% ---------------------------------------------------------------------------
%% Stessi punteggi e soglie di src/modello_generativo.regola_gravita.
%% Un test Python verifica che l'accordo sia totale su molti casi casuali.

punti_persone(confermate, 2).
punti_persone(incerte, 1).
punti_persone(assenti, 0).

punti_velocita(alta, 2).
punti_velocita(media, 1).
punti_velocita(bassa, 0).

soglia_critico(5).
soglia_alto(3).

% punteggio(Persone, Velocita, Sostanze, Vulnerabilita, NAdiacenti, Punteggio)
punteggio(Persone, Vel, Sost, Vuln, Adj, P) :-
    punti_persone(Persone, Pp),
    punti_velocita(Vel, Pv),
    ( Sost == si ->  Ps = 1 ; Ps = 0 ),
    ( Vuln == alta -> Pvu = 1 ; Pvu = 0 ),
    ( Adj >= 2 ->    Pa = 1 ; Pa = 0 ),
    P is Pp + Pv + Ps + Pvu + Pa.

gravita_da_punteggio(P, critico) :- soglia_critico(S), P >= S, !.
gravita_da_punteggio(P, alto)    :- soglia_alto(S),    P >= S, !.
gravita_da_punteggio(_, basso).

% gravita_con(I, Velocita, G): gravita dell'incidente I se la velocita del fuoco fosse Velocita.
gravita_con(I, Vel, G) :-
    incidente(I, _, _, Sost, Pers, _),
    registro(I, Vuln, Adj),
    punteggio(Pers, Vel, Sost, Vuln, Adj, P),
    gravita_da_punteggio(P, G).

% gravita(I, G): gravita con la velocita stimata caricata per lo scenario.
gravita(I, G) :-
    velocita_stimata(I, Vel),
    gravita_con(I, Vel, G).

%% ---------------------------------------------------------------------------
%% 3. Abduzione
%% ---------------------------------------------------------------------------
%% Data una gravita osservata, quali velocita del fuoco la spiegano?
%% La velocita e la variabile latente: e l'ipotesi che il sistema deve formulare.

ipotesi_velocita(I, G, Vel) :-
    member(Vel, [alta, media, bassa]),
    gravita_con(I, Vel, G).

spiegazioni_velocita(I, G, Lista) :-
    findall(V, ipotesi_velocita(I, G, V), Lista).

%% ---------------------------------------------------------------------------
%% 4. Propagazione del pericolo (ricorsione)
%% ---------------------------------------------------------------------------
%% collegato(A, B): esiste un cammino di edifici adiacenti da A a B.
%% La versione con lista dei visitati evita i loop sui cicli del registro.
%% La profondita della ricorsione e al massimo il numero di incidenti dello scenario.

collegato(A, B) :- collegato(A, B, [A]).

collegato(A, B, _) :-
    adiacenza(A, B).
collegato(A, B, Visitati) :-
    adiacenza(A, C),
    \+ memberchk(C, Visitati),
    collegato(C, B, [C|Visitati]).

% esposto(I): l'incidente I e collegato a un incidente critico diverso da se.
esposto(I) :-
    gravita(J, critico),
    J \== I,
    collegato(I, J).

% Gravita operativa: un incidente alto esposto a un critico viene trattato come critico.
% Non sostituisce la gravita del dataset, che resta il target del ML.
gravita_operativa(I, critico) :-
    gravita(I, alto),
    esposto(I),
    !.
gravita_operativa(I, G) :-
    gravita(I, G).

%% ---------------------------------------------------------------------------
%% 5. Vincoli di integrita sulle assegnazioni
%% ---------------------------------------------------------------------------
%% violazione(V) e vera per ogni violazione della proposta di dispatch.
%% Una proposta valida non deve produrre alcun risultato.

violazione(doppia_assegnazione(M)) :-
    assegnato(M, I1),
    assegnato(M, I2),
    I1 @< I2.
violazione(non_idoneo(M, I)) :-
    assegnato(M, I),
    \+ idoneo(M, I).
violazione(incidente_scoperto(I)) :-
    incidente(I, _, _, _, _, _),
    \+ assegnato(_, I).

%% ---------------------------------------------------------------------------
%% 6. Statistiche della KB (per la relazione)
%% ---------------------------------------------------------------------------

% kb_statistiche(-Predicati, -Regole, -Clausole): conteggi sulle clausole caricate da questo file.
kb_statistiche(NPred, NRegole, NClausole) :-
    findall(N-R,
            ( predicate_property(user:H, file(F)), file_base_name(F, 'kb.pl'),
              predicate_property(user:H, number_of_clauses(N)),
              N > 0,
              aggregate_all(count, (clause(user:H, B), B \== true), R) ),
            Coppie),
    length(Coppie, NPred),
    findall(N, member(N-_, Coppie), Ns), sum_list(Ns, NClausole),
    findall(R, member(_-R, Coppie), Rs), sum_list(Rs, NRegole).
