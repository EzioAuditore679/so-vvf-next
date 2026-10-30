"""Analisi di sensibilità.

1. Recupero delle CPT: quanto la rete appresa dal dataset si avvicina alle probabilità del
   modello generativo, con 1000 righe (dataset del progetto) e con 20 000 righe.
   Il modello generativo non è un esperto indipendente: è la fonte dei dati. Il test misura
   quanto bene lo stimatore recupera le probabilità dichiarate, non la loro correttezza.
2. Rumore delle etichette: gli esperimenti A, B, C e l'oracolo ripetuti con rumore 0.05, 0.12 e 0.20.

"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  
import pandas as pd  

import bayes_engine as be  
import config  
import generate_dataset as gd  
import modello_generativo as mg  
import train_models as tm  
from domain import Persone, SostanzePericolose, TipoEdificio, VelocitaFuoco  

RUMORI = [0.05, 0.12, 0.20]


def probabilita_vere() -> list[tuple]:
    """Ogni cella delle CPT della rete, con la probabilità del modello generativo.

    Ritorna tuple (nodo, stato, genitori, probabilità vera).
    """
    celle = []
    for tipo in TipoEdificio:
        celle.append(("tipo_edificio", tipo.value, {}, mg.P_TIPO[tipo]))
    celle.append(("meteo", "vento_forte", {}, mg.P_VENTO_FORTE))
    celle.append(("meteo", "calmo", {}, 1 - mg.P_VENTO_FORTE))
    for tipo in TipoEdificio:
        for sost in SostanzePericolose:
            p_si = mg.P_SOSTANZE_SI[tipo]
            celle.append(("sostanze_pericolose", sost.value, {"tipo_edificio": tipo.value},
                          p_si if sost is SostanzePericolose.SI else 1 - p_si))
        for piano in range(7):
            if tipo is TipoEdificio.BOSCO:
                p = 1.0 if piano == 0 else 0.0
            elif piano == 0:
                p = 0.0
            else:
                p = mg.PESI_PIANO[tipo][piano - 1]
            celle.append(("piano", str(piano), {"tipo_edificio": tipo.value}, p))
            distr = mg._distribuzione_persone(tipo, piano)
            for pers in Persone:
                celle.append(("persone", pers.value, {"tipo_edificio": tipo.value, "piano": str(piano)},
                              distr[pers]))
    for meteo in mg.Meteo:
        for sost in SostanzePericolose:
            distr = mg.P_VELOCITA[(meteo, sost)]
            for vel in VelocitaFuoco:
                celle.append(("velocita_fuoco", vel.value,
                              {"meteo": meteo.value, "sostanze_pericolose": sost.value}, distr[vel]))
    return celle


def errore_cpt(modello) -> float:
    """Errore assoluto medio tra le CPT apprese e quelle del modello generativo."""
    errori = []
    for nodo, stato, genitori, vera in probabilita_vere():
        stimata = be.probabilita(modello, nodo, stato, **genitori)
        errori.append(abs(stimata - vera))
    return float(np.mean(errori))


def recupero_cpt(seme: int = config.SEME) -> pd.DataFrame:
    righe = []
    for n in [config.N_TESTI_SINTETICI, 20_000]:
        df = pd.DataFrame(gd.genera(n, seme + n))
        modello = be.stima(df)
        righe.append(dict(n_righe=n, errore_medio_cpt=errore_cpt(modello)))
    return pd.DataFrame(righe)


def sensibilita_rumore(valori: list = RUMORI) -> pd.DataFrame:
    originale = config.RUMORE_GRAVITA
    tutte = []
    try:
        for r in valori:
            config.RUMORE_GRAVITA = r
            df = pd.DataFrame(gd.genera(config.N_TESTI_SINTETICI, config.SEME))
            for riga in tm.esegui(df):
                riga["rumore"] = r
                tutte.append(riga)
    finally:
        config.RUMORE_GRAVITA = originale
    esiti = pd.DataFrame(tutte)
    gruppi = esiti.groupby(["rumore", "esperimento", "modello"])
    return gruppi.agg(
        accuracy_media=("accuracy", "mean"), accuracy_dev=("accuracy", "std"),
        f1_media=("f1_macro", "mean"), f1_dev=("f1_macro", "std"),
    ).reset_index()


if __name__ == "__main__":
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cpt = recupero_cpt()
    cpt.to_csv(config.RESULTS_DIR / "sensitivita_cpt.csv", index=False)
    print(cpt.round(4).to_string(index=False))
    rum = sensibilita_rumore()
    rum.to_csv(config.RESULTS_DIR / "sensitivita_rumore.csv", index=False)
    print(rum.round(3).to_string(index=False))
