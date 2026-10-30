% knowledge_base/kb.pl: la knowledge base di SO-VVF-NEXT, in un solo file.
% Contiene la conoscenza di dominio (fatti), le regole e la dichiarazione dei fatti di scenario,
% che src/prolog_engine.py asserisce per ogni scenario (incidenti, mezzi, registro, stime).
% I fatti di dominio devono coincidere con le costanti di src/modello_generativo.py e src/domain.py:
% lo verifica tests/test_prolog.py.

% --- Conoscenza di dominio --------------------------------------------------

% Oltre questo piano serve l'autoscala per soccorrere le persone.
soglia_piani_autoscala(4).

% Regola di gravità: punteggio = persone + velocità + [sostanze] + [vulnerabilità alta] + [adiacenti >= 2].
punti_persone(confermate, 2).
punti_persone(incerte, 1).
punti_persone(assenti, 0).
punti_velocita(alta, 2).
punti_velocita(media, 1).
punti_velocita(bassa, 0).
soglia_critico(5).
soglia_alto(3).

% --- Fatti di scenario, asseriti da Python -----------------------------------

:- dynamic incidente/6.        % incidente(Id, Tipo, Piano, Sostanze, Persone, Meteo)
:- dynamic registro/3.         % registro(Id, Vulnerabilita, NAdiacenti)
:- dynamic adiacenza/2.        % adiacenza(IdA, IdB), caricata nei due versi
:- dynamic mezzo/2.            % mezzo(Id, TipoMezzo)
:- dynamic velocita_stimata/2. % velocita_stimata(Id, Velocita)
:- dynamic gravita_stimata/2.  % gravita_stimata(Id, Gravita): stima esterna (ML, rete), se presente
:- dynamic assegnato/3.        % assegnato(IdMezzo, IdIncidente, Ruolo)

% Il canadair non interviene dove ci sono sostanze pericolose (negazione per fallimento).
incompatibile(M, I) :-
    mezzo(M, canadair),
    incidente(I, _, _, si, _, _).

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

% Gravita di partenza: la stima esterna se c'e (ML o rete bayesiana), altrimenti la regola
% applicata alla velocita stimata del fuoco.
gravita_base(I, G) :-
    gravita_stimata(I, G0),
    !,
    G = G0.
gravita_base(I, G) :-
    gravita(I, G).

esposto(I) :-
    gravita_base(J, critico),
    J \== I,
    collegato(I, J).

% Gravita operativa: un incidente alto esposto a uno critico e trattato come critico.
% Non sostituisce la gravita del dataset, che resta il target del ML.
% Forma deterministica: corretta anche quando G e gia legata (ad esempio G = alto).
gravita_operativa(I, G) :-
    gravita_base(I, G0),
    (   G0 == alto, esposto(I)
    ->  G = critico
    ;   G = G0
    ).

% Composizione della squadra. Ogni ruolo richiesto da un incidente va coperto da un mezzo distinto.
ruolo(I, estinzione) :-
    incidente(I, _, _, _, _, _).
ruolo(I, rinforzo) :-
    gravita_operativa(I, critico).
ruolo(I, soccorso_quota) :-
    incidente(I, Tipo, Piano, _, Persone, _),
    Tipo \== bosco,
    soglia_piani_autoscala(S),
    Piano >= S,
    persone_da_soccorrere(I, Persone).
ruolo(I, decontaminazione) :-
    incidente(I, _, _, si, _, _).

% Persone incerte in quota: si manda l'autoscala solo se la gravita non e bassa.
persone_da_soccorrere(_, confermate).
persone_da_soccorrere(I, incerte) :-
    gravita_operativa(I, G),
    G \== basso.

squadra_richiesta(I, Ruoli) :-
    findall(R, ruolo(I, R), Ruoli).

% Quali tipi di mezzo possono coprire un ruolo.
adatto(autopompa, estinzione, _).
adatto(autopompa, rinforzo, _).
adatto(canadair, estinzione, I) :- incidente(I, bosco, _, _, _, _).
adatto(canadair, rinforzo, I) :- incidente(I, bosco, _, _, _, _).
adatto(autoscala, soccorso_quota, _).
adatto(nucleo_nbcr, decontaminazione, _).

copre(M, I, R) :-
    ruolo(I, R),
    mezzo(M, T),
    adatto(T, R, I),
    \+ incompatibile(M, I).

mezzi_adatti(I, R, Lista) :-
    findall(M, copre(M, I, R), Lista).

% Un mezzo e idoneo per un incidente se copre almeno uno dei suoi ruoli.
idoneo(M, I) :-
    mezzo(M, _),
    incidente(I, _, _, _, _, _),
    once(copre(M, I, _)).

mezzi_idonei(I, Lista) :-
    findall(M, idoneo(M, I), Lista).

% Spiegazioni: perche un ruolo e richiesto.
motivo(I, estinzione, incendio_segnalato) :-
    ruolo(I, estinzione).
motivo(I, rinforzo, gravita_critica) :-
    gravita_base(I, critico).
motivo(I, rinforzo, esposto_a(J)) :-
    gravita_base(I, alto),
    gravita_base(J, critico),
    J \== I,
    collegato(I, J).
motivo(I, soccorso_quota, persone_in_quota(Persone, Piano)) :-
    ruolo(I, soccorso_quota),
    incidente(I, _, Piano, _, Persone, _).
motivo(I, decontaminazione, sostanze_pericolose) :-
    ruolo(I, decontaminazione).

% Integrita delle assegnazioni: una proposta valida non produce violazioni.
% Un ruolo lasciato scoperto non e una violazione di integrita ma una richiesta di rinforzi,
% segnalata a parte da ruolo_scoperto/2.
violazione(doppia_assegnazione(M)) :-
    assegnato(M, I1, R1),
    assegnato(M, I2, R2),
    (I1, R1) @< (I2, R2).
violazione(non_adatto(M, I, R)) :-
    assegnato(M, I, R),
    \+ copre(M, I, R).
violazione(ruolo_coperto_due_volte(I, R)) :-
    assegnato(M1, I, R),
    assegnato(M2, I, R),
    M1 @< M2.

ruolo_scoperto(I, R) :-
    ruolo(I, R),
    \+ assegnato(_, I, R).

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
