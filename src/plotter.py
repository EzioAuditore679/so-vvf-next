"""Grafici dei risultati. Legge solo i CSV in results/ e scrive le immagini in plots/.

    python src/plotter.py

- ml_modelli.png:      accuratezza, F1 macro e F1 pesata per esperimento e modello (media ± dev. std.)
- ricerca_percorsi.png: nodi espansi da BFS, LcFS, A* Manhattan e A* PD; tempo in più della BFS
- assegnazione_bb.png: nodi visitati dal branch and bound, con e senza stima
- dispatch_fonti.png:  fonti della gravità valutate sul dispatch
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

# Palette usata
BLU, ARANCIO, VERDE, GIALLO = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRIGIO, GRIGIO_CHIARO = "#52514e", "#e4e3df"

NOMI_MODELLI = {
    "regressione_logistica": ("regressione logistica", BLU),
    "random_forest": ("random forest", ARANCIO),
    "naive_bayes": ("naive Bayes", VERDE),
    "rete_bayesiana": ("rete bayesiana", GIALLO),
}
ESPERIMENTI = ["A", "B", "rete", "C", "oracolo"]
METRICHE_ML = [("accuracy", "Accuratezza"), ("f1_macro", "F1 macro"), ("f1_pesata", "F1 pesata")]

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": GRIGIO_CHIARO,
    "axes.labelcolor": GRIGIO,
    "xtick.color": GRIGIO,
    "ytick.color": GRIGIO,
    "axes.titlesize": 10,
    "axes.titlelocation": "left",
})


def _salva(fig, nome: str) -> None:
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / nome, dpi=150)
    plt.close(fig)


def grafico_ml() -> None:
    """Una barra orizzontale per ogni coppia (esperimento, modello), raggruppate per esperimento."""
    ml = pd.read_csv(config.RESULTS_DIR / "ml_riepilogo.csv")
    baseline = ml[ml.modello == "baseline_maggioranza"].iloc[0]
    ml = ml[ml.modello.isin(NOMI_MODELLI)]

    righe, posizioni, etichette, y = [], [], [], 0
    for esperimento in ESPERIMENTI:
        for modello in NOMI_MODELLI:
            riga = ml[(ml.esperimento == esperimento) & (ml.modello == modello)]
            if riga.empty:
                continue
            righe.append(riga.iloc[0])
            posizioni.append(y)
            etichette.append(f"{esperimento} · {NOMI_MODELLI[modello][0]}")
            y += 1
        y += 0.6

    fig, assi = plt.subplots(1, len(METRICHE_ML), figsize=(13, 5.2), sharey=True)
    for ax, (metrica, titolo) in zip(assi, METRICHE_ML):
        medie = [r[f"{metrica}_media"] for r in righe]
        dev = [r[f"{metrica}_dev"] for r in righe]
        colori = [NOMI_MODELLI[r.modello][1] for r in righe]
        ax.barh(posizioni, medie, xerr=dev, color=colori, height=0.8, ecolor=GRIGIO, capsize=2)
        for p, m in zip(posizioni, medie):
            ax.text(0.02, p, f"{m:.2f}", va="center", color="white", fontsize=8)
        ax.axvline(baseline[f"{metrica}_media"], color=GRIGIO, linestyle="--", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_title(titolo)
        ax.xaxis.grid(True, color=GRIGIO_CHIARO)
        ax.set_axisbelow(True)
    assi[0].set_yticks(posizioni, etichette)
    assi[0].invert_yaxis()  # A in alto, oracolo in basso
    fig.suptitle("Classificazione della gravità: media ± dev. std. sui 5 fold esterni della nested CV "
                 "(linea tratteggiata: baseline a maggioranza)", x=0.01, ha="left", fontsize=10)
    _salva(fig, "ml_modelli.png")


def grafico_percorsi() -> None:
    """Ricerca del percorso: nodi espansi per algoritmo e costo in più della BFS, al variare delle chiusure."""
    ricerca = pd.read_csv(config.RESULTS_DIR / "ricerca_riepilogo.csv")
    chiusure = sorted(ricerca.chiusure.unique())
    etichette = [f"{100 * c:.0f}%" for c in chiusure]
    fig, (sx, dx) = plt.subplots(1, 2, figsize=(12, 4.2))

    for algoritmo, colore in [("BFS", GIALLO), ("LcFS", BLU), ("A* Manhattan", ARANCIO), ("A* PD", VERDE)]:
        dati = ricerca[ricerca.algoritmo == algoritmo]
        sx.plot(etichette, dati.espansi_media, marker="o", color=colore, linewidth=2, label=algoritmo)
    sx.set_ylim(bottom=0)
    sx.set_xlabel("celle libere chiuse temporaneamente")
    sx.set_ylabel(f"nodi espansi per percorso (media su {config.N_MAPPE * config.N_MEZZI_MAPPA})")
    sx.set_title("Nodi espansi")
    sx.legend(frameon=False)

    bfs = ricerca[ricerca.algoritmo == "BFS"]
    sx.yaxis.grid(True, color=GRIGIO_CHIARO)
    dx.bar(etichette, bfs.costo_in_piu_pct_media, color=GIALLO, width=0.6)
    for x, (v, ottimi) in enumerate(zip(bfs.costo_in_piu_pct_media, bfs.ottimi_pct)):
        dx.text(x, v, f"{v:.1f}%\n(ottima nel {ottimi:.0f}%)", ha="center", va="bottom", color=GRIGIO, fontsize=8)
    dx.set_xlabel("celle libere chiuse temporaneamente")
    dx.set_ylabel("tempo in più rispetto all'ottimo (%)")
    dx.set_title("BFS: tempo in più del percorso trovato")
    dx.set_ylim(0, bfs.costo_in_piu_pct_media.max() * 1.35)
    dx.yaxis.grid(True, color=GRIGIO_CHIARO)
    dx.set_axisbelow(True)
    fig.suptitle(f"Percorso del mezzo su mappe {config.DIM_GRIGLIA[0]}×{config.DIM_GRIGLIA[1]}: "
                 "LcFS e A* trovano sempre l'ottimo", x=0.01, ha="left", fontsize=10)
    _salva(fig, "ricerca_percorsi.png")


def grafico_assegnazione() -> None:
    """Branch and bound per l'assegnazione dei mezzi: nodi visitati con e senza stima."""
    dati = pd.read_csv(config.RESULTS_DIR / "dispatch_riepilogo.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for algoritmo, colore in [("senza stima", BLU), ("con stima", VERDE)]:
        d = dati[dati.algoritmo == algoritmo]
        ax.plot(d.incidenti, d.nodi_media, marker="o", color=colore, linewidth=2, label=algoritmo)
        for n, nodi, completi, casi in zip(d.incidenti, d.nodi_media, d.completi, d.casi):
            if completi < casi:
                ax.annotate(f"{completi}/{casi}", (n, nodi), xytext=(6, -10), textcoords="offset points",
                            fontsize=8, color=colore)
    ax.set_yscale("log")
    ax.set_xticks(sorted(dati.incidenti.unique()))
    ax.set_xlabel("incidenti per scenario")
    ax.set_ylabel(f"nodi visitati (media su {config.N_SCENARI_PER_DIMENSIONE} scenari)")
    ax.set_title(f"Assegnazione dei mezzi: branch and bound\n(x/y: casi chiusi entro {config.LIMITE_NODI:,} nodi)"
                 .replace(",", " "))
    ax.yaxis.grid(True, color=GRIGIO_CHIARO)
    ax.legend(frameon=False)
    _salva(fig, "assegnazione_bb.png")


def grafico_dispatch() -> None:
    """Per ogni fonte della gravità: critici mancati (totale), sovrastimati e costo in più (media con IC 95%)."""
    scenari = pd.read_csv(config.RESULTS_DIR / "dispatch_e2e_scenari.csv")
    fonti = [("ml", "ML", BLU), ("rete", "rete", ARANCIO), ("rete_kb", "rete + KB", VERDE),
             ("prudente", "prudente", GIALLO)]
    critici_veri = int(scenari[scenari.fonte == "vera"].critici_veri.sum())

    fig, assi = plt.subplots(1, 3, figsize=(12, 3.8))
    pannelli = [
        ("critici_mancati", f"Incidenti critici mancati (totale su {critici_veri})"),
        ("sovrastimati", "Incidenti sovrastimati per scenario"),
        ("regret", "Costo in più rispetto all'ottimo"),
    ]
    for ax, (colonna, titolo) in zip(assi, pannelli):
        valori, errori = [], []
        for fonte, _, _ in fonti:
            x = scenari[scenari.fonte == fonte][colonna]
            if colonna == "critici_mancati":
                valori.append(x.sum())
                errori.append(0)
            else:
                # Intervallo di confidenza al 95% della media: ±1.96 · dev. std. / √n
                valori.append(x.mean())
                errori.append(1.96 * x.std() / np.sqrt(len(x)))
        posizioni = range(len(fonti))
        barre_errore = errori if colonna != "critici_mancati" else None
        ax.bar(posizioni, valori, yerr=barre_errore, color=[c for _, _, c in fonti], width=0.6,
               ecolor=GRIGIO, capsize=3)
        for p, v, e in zip(posizioni, valori, errori):
            ax.text(p, v + e, f"{v:.0f}" if colonna == "critici_mancati" else f"{v:.2f}",
                    ha="center", va="bottom", color=GRIGIO, fontsize=8)
        ax.set_xticks(posizioni, [nome for _, nome, _ in fonti])
        ax.set_title(titolo)
        ax.yaxis.grid(True, color=GRIGIO_CHIARO)
        ax.set_axisbelow(True)
    fig.suptitle(f"Fonti della gravità valutate sul dispatch: {config.N_SCENARI_E2E} scenari da "
                 f"{config.N_INCIDENTI_E2E} incidenti (barre: IC 95% della media)", x=0.01, ha="left", fontsize=10)
    _salva(fig, "dispatch_fonti.png")


def tutti() -> None:
    grafico_ml()
    grafico_percorsi()
    grafico_assegnazione()
    grafico_dispatch()
    print(f"Grafici scritti in {config.PLOTS_DIR}")


if __name__ == "__main__":
    tutti()
