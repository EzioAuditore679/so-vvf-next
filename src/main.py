"""SO-VVF-NEXT: demo integrata e pipeline degli esperimenti.

    python src/main.py                                  demo su quattro chiamate (pochi secondi)
    python src/main.py esperimenti                      rigenera dataset, test, risultati e grafici (~15 min)
    python src/main.py esperimenti --senza-sensibilita  come sopra, senza l'analisi di sensibilità (~5 min)
    python src/main.py grafici                          solo i grafici (src/plotter.py), dai CSV in results/

La demo percorre l'intera catena: testo della chiamata → campi (NLP) → gravità stimata da ML, rete
bayesiana e rete + regola della KB → squadra richiesta con le spiegazioni Prolog → assegnazione dei mezzi
(CSP, branch and bound) → percorsi (BFS, LcFS, A*) → verifica di integrità della proposta. Il dispatch usa la gravità della fonte rete + KB, che nella
valutazione end-to-end manca meno incidenti critici (results/dispatch_e2e_riepilogo.csv).
"""

import argparse
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

import config
import plotter


# Quattro chiamate pensate per mostrare i casi interessanti. c2 e c3 sono vicine: se c2 è critica,
# c3 (alta) diventa critica per propagazione. Il registro della zona dà vulnerabilità e adiacenti.
CHIAMATE = [
    dict(id="c1", cella=(3, 4), vulnerabilita="bassa", adiacenti=1,
         testo="Pronto, c'è un incendio in una palazzina al quinto piano. Mi dicono che ci sono persone "
               "intrappolate. C'è vento forte."),
    dict(id="c2", cella=(10, 10), vulnerabilita="alta", adiacenti=2,
         testo="Salve, abbiamo un incendio in un capannone al primo piano. Vicino alle fiamme ci sono "
               "bombole di gas. Mi dicono che c'è qualcuno bloccato. Il vento è forte."),
    dict(id="c3", cella=(11, 12), vulnerabilita="bassa", adiacenti=1,
         testo="Segnalo un incendio in un deposito al secondo piano. Niente sostanze pericolose. "
               "Mi dicono che qualcuno non riesce a uscire. C'è vento forte."),
    dict(id="c4", cella=(17, 17), vulnerabilita="bassa", adiacenti=0,
         testo="Buongiorno, c'è un incendio in una pineta. Nessuna sostanza pericolosa. "
               "Mi dicono che tutti sono usciti. Non c'è vento."),
]

FLOTTA = {
    "aps1": ("autopompa", (2, 2)), "aps2": ("autopompa", (12, 8)), "aps3": ("autopompa", (8, 15)),
    "aps4": ("autopompa", (18, 3)), "as1": ("autoscala", (5, 1)), "nbcr1": ("nucleo_nbcr", (14, 10)),
    "cnd1": ("canadair", (19, 19)),
}


def _mappa_demo():
    """Mappa generata dal seme, con le celle di incidenti e stazioni libere."""
    import ricerca

    m = ricerca.genera_mappa(random.Random(config.SEME))
    occupate = {c["cella"] for c in CHIAMATE} | {cella for _, cella in FLOTTA.values()}
    return ricerca.Mappa(m.righe, m.colonne, m.ostacoli - occupate, m.lente, m.chiusure - occupate)


def demo() -> None:
    import bayes_engine as be
    import dispatch as dp
    import ricerca
    import valutazione_dispatch as vd
    from domain import TipoMezzo
    from nlp_extract import estrai_campi
    from prolog_engine import KnowledgeBase

    pd.set_option("display.width", 160)
    df = pd.read_csv(config.DATA_DIR / "dataset_sintetico.csv")
    print("Addestramento di regressione logistica e rete bayesiana sul dataset sintetico...")
    stimatori = vd.addestra_stimatori(df)

    print("\n1. Campi estratti dal testo delle chiamate")
    campi = {c["id"]: estrai_campi(c["testo"]) for c in CHIAMATE}
    for c in CHIAMATE:
        print(f"  {c['id']}: {c['testo']}")
        print(f"      → {campi[c['id']]}  registro: vulnerabilità {c['vulnerabilita']}, "
              f"adiacenti {c['adiacenti']}")

    ids = [c["id"] for c in CHIAMATE]
    ev = pd.DataFrame([campi[i] for i in ids], index=ids).fillna("nessuno")
    ev["piano"] = ev["piano"].astype(str)
    ev["vulnerabilita"] = [c["vulnerabilita"] for c in CHIAMATE]
    ev["adiacenti"] = [str(c["adiacenti"]) for c in CHIAMATE]

    ml = stimatori["ml"].predict(ev[stimatori["colonne_ml"]])
    p = be.distribuzione_gravita(stimatori["rete"], ev, be.OSSERVATE_CON_REGISTRO)
    rete = be.gravita_piu_probabile(stimatori["rete"], ev, be.OSSERVATE_CON_REGISTRO)
    vel = dict(zip(ids, be.velocita_piu_probabile(stimatori["rete"], ev, be.OSSERVATE)))

    kb = KnowledgeBase()
    mappa = _mappa_demo()
    caso = dp.Caso(mappa, {m: (TipoMezzo(t), cella) for m, (t, cella) in FLOTTA.items()},
                  {c["id"]: (None, c["cella"]) for c in CHIAMATE})
    incidenti = [dict(id=i, tipo=campi[i]["tipo_edificio"], piano=campi[i]["piano"] or 0,
                      sostanze=campi[i]["sostanze_pericolose"], persone=campi[i]["persone"],
                      meteo=campi[i]["meteo"]) for i in ids]
    registro = {c["id"]: (c["vulnerabilita"], c["adiacenti"]) for c in CHIAMATE}
    adiacenze = dp.adiacenze_caso(caso)
    mezzi = {m: t for m, (t, _) in FLOTTA.items()}

    def carica(velocita):
        kb.carica_scenario(incidenti, mezzi, registro, adiacenze, velocita)

    carica({i: "alta" for i in ids})
    prudente = {i: kb.gravita(i) for i in ids}
    carica(vel)  # fonte usata per il dispatch: velocità stimata dalla rete, regola della KB

    print("\n2. Gravità stimata dalle diverse fonti")
    tab = pd.DataFrame({
        "ML": ml, "rete": rete, "P(critico) rete": p[:, be.STATI["gravita"].index("critico")].round(2),
        "velocità (rete)": [vel[i] for i in ids], "rete + KB": [kb.gravita(i) for i in ids],
        "prudente": [prudente[i] for i in ids], "operativa (rete + KB)": [kb.gravita_operativa(i) for i in ids],
    }, index=ids)
    print(tab.to_string())
    print(f"  Incidenti adiacenti (distanza ≤ {config.DISTANZA_ADIACENZA}): {adiacenze or 'nessuno'}")
    for i, g in zip(ids, ml):
        compatibili = kb.spiegazioni_velocita(i, g)
        print(f"  Abduzione {i}: con la gravità del ML ({g}) la velocità del fuoco può essere "
              f"{', '.join(compatibili) or 'nessuna: la stima del ML è incoerente con la regola'}")

    print("\n3. Squadra richiesta dalla KB, con le spiegazioni")
    adatti = {}
    for i in ids:
        for ruolo, motivo in kb.motivi(i):
            adatti[(i, ruolo)] = kb.mezzi_adatti(i, ruolo)
            print(f"  {i} {ruolo:16s} perché {motivo:32s} mezzi adatti: {', '.join(adatti[(i, ruolo)]) or '-'}")

    print("\n4. Assegnazione dei mezzi: CSP con branch and bound")
    sc = dp.scenario_da_adatti(adatti, tuple(FLOTTA), dp.costi_caso(caso))
    soluzione = dp.risolvi(sc)
    senza_stima = dp.risolvi(sc, usa_stima=False)
    for m, i, ruolo in sorted(soluzione.assegnazioni, key=lambda a: (a[1], a[2])):
        print(f"  {m:6s} ({FLOTTA[m][0]:12s}) → {i} {ruolo:16s} tempo {sc.costo[(m, i)]}")
    for i, ruolo in soluzione.scoperti:
        print(f"  ruolo scoperto: {i} {ruolo} → richiesta di rinforzi (penalità {sc.penalita})")
    print(f"  Costo totale {soluzione.costo}; nodi visitati: con stima {soluzione.nodi}, "
          f"senza stima {senza_stima.nodi} (stesso costo: {senza_stima.costo == soluzione.costo})")

    print("\n5. Percorsi dei mezzi assegnati: nodi espansi da ogni algoritmo")
    print(f"  {'mezzo':6s} {'incidente':9s} {'tempo':>5s} {'BFS':>14s} {'LcFS':>6s} {'A* Manhattan':>13s} {'A* PD':>6s}")
    for m, i, _ in sorted(soluzione.assegnazioni, key=lambda a: (a[1], a[0])):
        partenza, arrivo = FLOTTA[m][1], dict((c["id"], c["cella"]) for c in CHIAMATE)[i]
        tabella, espansi_pd = ricerca.cost_to_goal(mappa, arrivo)
        obiettivo = lambda c, arrivo=arrivo: c == arrivo
        b = ricerca.bfs(partenza, obiettivo, mappa.successori)
        e = {nome: ricerca.a_star(partenza, obiettivo, mappa.successori, h).espansi
             for nome, h in [("LcFS", lambda c: 0), ("M", ricerca.h_manhattan(arrivo)), ("PD", ricerca.h_pd(tabella))]}
        bfs_testo = f"{b.espansi} (tempo {b.costo:g})"
        print(f"  {m:6s} {i:9s} {sc.costo[(m, i)]:>5g} {bfs_testo:>14s} {e['LcFS']:>6} {e['M']:>13} {e['PD']:>6}")
    print(f"  La tabella della PD costa {espansi_pd} nodi per incidente (sulla mappa senza chiusure) "
          f"ed è riusata da tutti i mezzi diretti a quell'incidente.")

    violazioni = kb.violazioni(soluzione.assegnazioni)
    print("\n6. Verifica di integrità in Prolog:", "proposta valida" if not violazioni else violazioni)


# --- Pipeline ------------------------------------------------------------------------------------

def _esegui(*argomenti: str) -> None:
    print(f"\n$ python {' '.join(argomenti)}", flush=True)
    subprocess.run([sys.executable, *argomenti], cwd=ROOT, check=True)


def esperimenti(sensibilita: bool = True) -> None:
    _esegui("src/generate_dataset.py")
    _esegui("-m", "pytest", "-q", "tests")
    _esegui("src/ricerca.py")
    _esegui("src/dispatch.py")
    _esegui("src/train_models.py")
    if sensibilita:
        _esegui("src/sensitivita.py")
    _esegui("src/valutazione_dispatch.py")
    plotter.tutti()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("comando", nargs="?", default="demo", choices=["demo", "esperimenti", "grafici"])
    parser.add_argument("--senza-sensibilita", action="store_true", help="salta src/sensitivita.py")
    args = parser.parse_args()
    if args.comando == "demo":
        demo()
    elif args.comando == "esperimenti":
        esperimenti(not args.senza_sensibilita)
    else:
        plotter.tutti()
