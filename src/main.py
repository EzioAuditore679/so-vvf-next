"""SO-VVF-NEXT: demo integrata e pipeline degli esperimenti.

    python src/main.py                                  demo su quattro chiamate (pochi secondi)
    python src/main.py esperimenti                      rigenera dataset, test, risultati e grafici (~15 min)
    python src/main.py esperimenti --senza-sensibilita  come sopra, senza l'analisi di sensibilità (~5 min)
    python src/main.py grafici                          solo i grafici, dai CSV in results/

La demo percorre l'intera catena: testo della chiamata → campi (NLP) → gravità stimata da ML, rete
bayesiana e rete + regola della KB → squadra richiesta con le spiegazioni Prolog → dispatch con A*
→ verifica di integrità della proposta. Il dispatch usa la gravità della fonte rete + KB, che nella
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


def _griglia_demo():
    import search as s

    g = s.genera_griglia(random.Random(config.SEME), *config.DIM_GRIGLIA, config.PROB_OSTACOLO)
    occupate = {c["cella"] for c in CHIAMATE} | {cella for _, cella in FLOTTA.values()}
    return s.Griglia(g.righe, g.colonne, g.ostacoli - occupate)


def demo() -> None:
    import bayes_engine as be
    import search as s
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
    griglia = _griglia_demo()
    caso = s.Caso(griglia, {m: (TipoMezzo(t), cella) for m, (t, cella) in FLOTTA.items()},
                  {c["id"]: (None, c["cella"]) for c in CHIAMATE})
    incidenti = [dict(id=i, tipo=campi[i]["tipo_edificio"], piano=campi[i]["piano"] or 0,
                      sostanze=campi[i]["sostanze_pericolose"], persone=campi[i]["persone"],
                      meteo=campi[i]["meteo"]) for i in ids]
    registro = {c["id"]: (c["vulnerabilita"], c["adiacenti"]) for c in CHIAMATE}
    adiacenze = s.adiacenze_caso(caso)
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

    print("\n4. Dispatch (A* con h_libero; LcFS per confronto)")
    sc = s.scenario_da_adatti(adatti, tuple(FLOTTA), s.costi_caso(caso))
    problema = s.Dispatch(sc)
    r = s.ricerca(problema.inizio(), problema.obiettivo, problema.successori, problema.h_libero)
    r0 = s.ricerca(problema.inizio(), problema.obiettivo, problema.successori, lambda st: 0)
    assegnazioni, scoperti = problema.piano(r.cammino)
    for m, i, ruolo in sorted(assegnazioni, key=lambda a: (a[1], a[2])):
        print(f"  {m:6s} ({FLOTTA[m][0]:12s}) → {i} {ruolo:16s} tempo {sc.costo[(m, i)]}")
    for i, ruolo in scoperti:
        print(f"  ruolo scoperto: {i} {ruolo} → richiesta di rinforzi (penalità {sc.penalita})")
    print(f"  Costo totale {r.costo}; nodi espansi: A* {r.espansi}, LcFS {r0.espansi} (stesso costo: {r0.costo == r.costo})")

    violazioni = kb.violazioni(assegnazioni)
    print("\n5. Verifica di integrità in Prolog:", "proposta valida" if not violazioni else violazioni)


# --- Grafici ---

# Palette usata
COLORI = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
TESTO, TESTO_2, GRIGLIA, FONDO = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _stile(ax) -> None:
    ax.set_facecolor(FONDO)
    for lato in ("top", "right"):
        ax.spines[lato].set_visible(False)
    for lato in ("left", "bottom"):
        ax.spines[lato].set_color(GRIGLIA)
    ax.tick_params(colors=TESTO_2, labelsize=9)
    ax.yaxis.grid(True, color=GRIGLIA, linewidth=0.8)
    ax.set_axisbelow(True)


def grafici() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "text.color": TESTO, "axes.labelcolor": TESTO_2,
                         "figure.facecolor": FONDO})

    # 1. Modelli: media ± dev. std. su 5 fold esterni, per esperimento.
    ml = pd.read_csv(config.RESULTS_DIR / "ml_riepilogo.csv")
    base = ml[ml.modello == "baseline_maggioranza"].iloc[0]
    modelli = ["regressione_logistica", "random_forest", "naive_bayes", "rete_bayesiana"]
    nomi = {"regressione_logistica": "regressione logistica", "random_forest": "random forest",
            "naive_bayes": "naive Bayes", "rete_bayesiana": "rete bayesiana"}
    esperimenti = ["A", "B", "rete", "C", "oracolo"]
    fig, assi = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    larghezza = 0.2
    for ax, (metrica, titolo) in zip(assi, [("f1", "Macro F1"), ("accuracy", "Accuratezza")]):
        _stile(ax)
        for e, esp in enumerate(esperimenti):
            presenti = [m for m in modelli if ((ml.esperimento == esp) & (ml.modello == m)).any()]
            for k, mod in enumerate(presenti):
                riga = ml[(ml.esperimento == esp) & (ml.modello == mod)].iloc[0]
                x = e + (k - (len(presenti) - 1) / 2) * larghezza
                v = riga[f"{metrica}_media"]
                ax.bar(x, v, larghezza * 0.9, color=COLORI[modelli.index(mod)], yerr=riga[f"{metrica}_dev"],
                       ecolor=TESTO_2, capsize=2, label=nomi[mod] if e == esperimenti.index(
                           ml[ml.modello == mod].esperimento.iloc[0]) else None)
                ax.text(x, 0.03, f"{v:.2f}", rotation=90, ha="center", va="bottom", fontsize=7, color="white")
        ax.axhline(base[f"{metrica}_media"], color=TESTO_2, linestyle="--", linewidth=1,
                   label="baseline a maggioranza")
        ax.set_xticks(np.arange(len(esperimenti)), esperimenti)
        ax.set_ylim(0, 1)
        ax.set_title(f"{titolo} (media ± dev. std., nested CV 5×5)", fontsize=10, loc="left")
    assi[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "ml_modelli.png", dpi=150)
    plt.close(fig)

    # 2. Ricerca: nodi espansi in funzione del numero di incidenti.
    ric = pd.read_csv(config.RESULTS_DIR / "search_riepilogo.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    _stile(ax)
    for k, alg in enumerate(["LcFS", "A* h_min", "A* h_libero"]):
        sub = ric[ric.algoritmo == alg]
        ax.plot(sub.dimensione, sub.espansi_media, color=COLORI[k], linewidth=2, marker="o", markersize=7,
                markeredgecolor=FONDO, markeredgewidth=2, label=alg)
        ultimo = sub.iloc[-1]
        ax.annotate(f"{ultimo.espansi_media:,.0f}".replace(",", " "), (ultimo.dimensione, ultimo.espansi_media),
                    textcoords="offset points", xytext=(6, 0), va="center", fontsize=8, color=TESTO_2)
    ax.set_yscale("log")
    ax.set_xlabel("incidenti per scenario")
    ax.set_ylabel("nodi espansi (media su 20 scenari)")
    ax.set_title("Dispatch per ruoli: LcFS e A*", fontsize=10, loc="left")
    ax.set_xticks(sorted(ric.dimensione.unique()))
    ax.set_xlim(right=ric.dimensione.max() + 1.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "ricerca_espansi.png", dpi=150)
    plt.close(fig)

    # 3. Fonti della gravità valutate sul dispatch (la fonte "vera" è il riferimento a zero).
    fonti = ["ml", "rete", "rete_kb", "prudente"]
    etichette = {"ml": "ML", "rete": "rete", "rete_kb": "rete + KB", "prudente": "prudente"}
    scenari = pd.read_csv(config.RESULTS_DIR / "dispatch_e2e_scenari.csv")
    rng = np.random.default_rng(config.SEME)

    def media_ic(fonte: str, col: str) -> tuple[float, float, float]:
        """Media e intervallo di confidenza al 95% della media (bootstrap sugli scenari)."""
        x = scenari.loc[scenari.fonte == fonte, col].to_numpy(dtype=float)
        medie = rng.choice(x, (2000, len(x))).mean(axis=1)
        return x.mean(), *np.percentile(medie, [2.5, 97.5])

    pannelli = [
        ("critici_mancati", "somma", "Incidenti critici mancati (su {n})"),
        ("sovrastimati", "media", "Incidenti sovrastimati per scenario"),
        ("regret", "media", "Costo in più rispetto all'ottimo"),
    ]
    fig, assi = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (col, tipo, titolo) in zip(assi, pannelli):
        _stile(ax)
        if tipo == "somma":
            valori = [scenari.loc[scenari.fonte == f, col].sum() for f in fonti]
            errori = None
        else:
            stime = [media_ic(f, col) for f in fonti]
            valori = [m for m, _, _ in stime]
            errori = np.array([[m - lo for m, lo, _ in stime], [hi - m for m, _, hi in stime]])
        ax.bar(range(len(fonti)), valori, 0.6, color=COLORI[:len(fonti)], yerr=errori, ecolor=TESTO_2, capsize=3)
        for x, v in enumerate(valori):
            testo = f"{v:.0f}" if tipo == "somma" else f"{v:.2f}"
            ax.text(x + (0.08 if errori is not None else 0), v, testo, ha="left" if errori is not None else "center",
                    va="bottom", fontsize=8, color=TESTO)
        ax.set_xticks(range(len(fonti)), [etichette[f] for f in fonti])
        n = int(scenari.loc[scenari.fonte == "ml", "critici_veri"].sum())
        ax.set_title(titolo.format(n=n), fontsize=10, loc="left")
    fig.suptitle(f"Fonti della gravità valutate sul dispatch ({config.N_SCENARI_E2E} scenari da "
                 f"{config.N_INCIDENTI_E2E} incidenti; media e IC 95% bootstrap della media)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "dispatch_fonti.png", dpi=150)
    plt.close(fig)
    print(f"Grafici scritti in {config.PLOTS_DIR}")


# --- Pipeline ------------------------------------------------------------------------------------

def _esegui(*argomenti: str) -> None:
    print(f"\n$ python {' '.join(argomenti)}", flush=True)
    subprocess.run([sys.executable, *argomenti], cwd=ROOT, check=True)


def esperimenti(sensibilita: bool = True) -> None:
    _esegui("src/generate_dataset.py")
    _esegui("-m", "pytest", "-q", "tests")
    _esegui("src/search.py")
    _esegui("src/train_models.py")
    if sensibilita:
        _esegui("src/sensitivita.py")
    _esegui("src/valutazione_dispatch.py")
    grafici()


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
        grafici()
