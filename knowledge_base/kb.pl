% knowledge_base/kb.pl
% Regole fisse. Soglie e punteggi sono in kb_statica.pl, generato da src/prolog_export.py.
% I fatti di scenario sono asseriti da src/prolog_engine.py.

:- include('kb_statica.pl').

:- dynamic incidente/6.        % incidente(Id, Tipo, Piano, Sostanze, Persone, Meteo)
:- dynamic registro/3.         % registro(Id, Vulnerabilita, NAdiacenti)
:- dynamic adiacenza/2.        % adiacenza(IdA, IdB), caricata nei due versi
:- dynamic mezzo/2.            % mezzo(Id, TipoMezzo)
:- dynamic velocita_stimata/2. % velocita_stimata(Id, Velocita)
:- dynamic assegnato/2.        % assegnato(IdMezzo, IdIncidente)

% Idoneita dei mezzi.
requisito(autopompa, _, _, _).
requisito(autoscala, Tipo, Piano, _) :-
    Tipo \== bosco,
    soglia_piani_autoscala(S),
    Piano >= S.
requisito(nucleo_nbcr, _, _, si).
requisito(canadair, bosco, _, _).

% Il canadair non interviene dove ci sono sostanze pericolose (negazione per fallimento).
incompatibile(M, I) :-
    mezzo(M, canadair),
    incidente(I, _, _, si, _, _).

idoneo(M, I) :-
    mezzo(M, T),
    incidente(I, Tipo, Piano, Sost, _, _),
    requisito(T, Tipo, Piano, Sost),
    \+ incompatibile(M, I).

mezzi_idonei(I, Lista) :-
    findall(M, idoneo(M, I), Lista).

% Gravita: stessi punteggi e soglie di modello_generativo.regola_gravita.
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

gravita_con(I, Vel, G) :-
    incidente(I, _, _, Sost, Pers, _),
    registro(I, Vuln, Adj),
    punteggio(Pers, Vel, Sost, Vuln, Adj, P),
    gravita_da_punteggio(P, G).

gravita(I, G) :-
    velocita_stimata(I, Vel),
    gravita_con(I, Vel, G).

% Abduzione: velocita del fuoco compatibili con una gravita osservata.
ipotesi_velocita(I, G, Vel) :-
    member(Vel, [alta, media, bassa]),
    gravita_con(I, Vel, G).

spiegazioni_velocita(I, G, Lista) :-
    findall(V, ipotesi_velocita(I, G, V), Lista).

% Propagazione del pericolo. La lista dei visitati impedisce i loop sui cicli.
collegato(A, B) :- collegato(A, B, [A]).

collegato(A, B, _) :-
    adiacenza(A, B).
collegato(A, B, Visitati) :-
    adiacenza(A, C),
    \+ memberchk(C, Visitati),
    collegato(C, B, [C|Visitati]).

esposto(I) :-
    gravita(J, critico),
    J \== I,
    collegato(I, J).

% Gravita operativa: un incidente alto esposto a uno critico e trattato come critico.
% Non sostituisce la gravita del dataset, che resta il target del ML.
gravita_operativa(I, critico) :-
    gravita(I, alto),
    esposto(I),
    !.
gravita_operativa(I, G) :-
    gravita(I, G).

% Integrita delle assegnazioni: una proposta valida non produce violazioni.
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

% Conteggi delle clausole di questo file, per la relazione.
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
