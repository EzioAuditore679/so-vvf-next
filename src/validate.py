"""Validazione del gold set e dell'estrattore NLP.

1. Controllo di integrità del file gold_set.csv: colonne, valori ammessi, id unici.
2. Valutazione di estrai_campi() sul gold set: accuratezza, precisione, recall,
   F1 macro e matrice di confusione per ogni campo.

Il gold set NON deve servire a scrivere o tarare le regole di nlp_extract.py:
se si correggono le regole guardando i suoi errori, il risultato diventa circolare.

Esecuzione dalla radice del progetto:
    python src/validate.py
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
from domain import Meteo, Persone, SostanzePericolose, TipoEdificio  # noqa: E402
from nlp_extract import estrai_campi  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

COLONNE = ["id", "testo", "tipo_edificio", "piano", "sostanze_pericolose", "persone", "meteo"]

VALORI_AMMESSI = {
    "tipo_edificio": {e.value for e in TipoEdificio},
    "sostanze_pericolose": {e.value for e in SostanzePericolose},
    "persone": {e.value for e in Persone},
    "meteo": {e.value for e in Meteo},
}
CAMPI_CATEGORICI = list(VALORI_AMMESSI)
ETICHETTA_ASSENTE = "nessuno"  # quando l'estrattore non riconosce il campo


def leggi(percorso: Path) -> list[dict]:
    with percorso.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def controlla_integrita(righe: list[dict]) -> list[str]:
    errori = []
    if not righe:
        return ["file vuoto"]
    if set(righe[0].keys()) != set(COLONNE):
        errori.append(f"colonne attese {COLONNE}, trovate {list(righe[0].keys())}")
    ids = [r["id"] for r in righe]
    if len(ids) != len(set(ids)):
        errori.append("id duplicati")
    for r in righe:
        if not r["testo"].strip():
            errori.append(f"{r['id']}: testo vuoto")
        if not r["piano"].isdigit() or int(r["piano"]) > 6:
            errori.append(f"{r['id']}: piano={r['piano']!r} non valido")
        for campo, ammessi in VALORI_AMMESSI.items():
            if r[campo] not in ammessi:
                errori.append(f"{r['id']}: {campo}={r[campo]!r} non ammesso")
    return errori


def valuta(righe: list[dict]) -> tuple[dict, list[dict]]:
    predizioni = [estrai_campi(r["testo"]) for r in righe]
    risultati = {}
    errori = []

    # Piano: valore intero, quindi basta l'accuratezza esatta.
    ok = [int(p["piano"]) == int(r["piano"]) for p, r in zip(predizioni, righe)]
    risultati["piano"] = {"accuracy": sum(ok) / len(ok)}
    for p, r, corretto in zip(predizioni, righe, ok):
        if not corretto:
            errori.append({"id": r["id"], "campo": "piano", "vero": r["piano"],
                           "predetto": p["piano"], "testo": r["testo"]})

    for campo in CAMPI_CATEGORICI:
        veri = [r[campo] for r in righe]
        pred = [p[campo] if p[campo] is not None else ETICHETTA_ASSENTE for p in predizioni]
        etichette = sorted(VALORI_AMMESSI[campo])
        precisione, recall, f1, _ = precision_recall_fscore_support(
            veri, pred, labels=etichette, average="macro", zero_division=0
        )
        matrice = confusion_matrix(veri, pred, labels=etichette + [ETICHETTA_ASSENTE])
        risultati[campo] = {
            "accuracy": accuracy_score(veri, pred),
            "precision_macro": precisione,
            "recall_macro": recall,
            "f1_macro": f1,
            "matrice_confusione": {
                "etichette": etichette + [ETICHETTA_ASSENTE],
                "matrice": matrice.tolist(),
            },
        }
        for p, r, v, q in zip(predizioni, righe, veri, pred):
            if v != q:
                errori.append({"id": r["id"], "campo": campo, "vero": v,
                               "predetto": q, "testo": r["testo"]})

    return risultati, errori


if __name__ == "__main__":
    righe = leggi(config.DATA_DIR / "gold_set.csv")

    problemi = controlla_integrita(righe)
    if problemi:
        print("Errori di integrità nel gold set:")
        for p in problemi[:20]:
            print(" -", p)
        sys.exit(1)
    print(f"Integrità OK: {len(righe)} testi.")

    risultati, errori = valuta(righe)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (config.RESULTS_DIR / "metrics_nlp_gold.json").open("w", encoding="utf-8") as f:
        json.dump(risultati, f, ensure_ascii=False, indent=2)
    with (config.RESULTS_DIR / "errori_nlp_gold.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "campo", "vero", "predetto", "testo"])
        writer.writeheader()
        writer.writerows(errori)

    print(f"\n{'campo':22s} {'accuracy':>9s} {'F1 macro':>9s}")
    for campo, m in risultati.items():
        f1 = m.get("f1_macro", float("nan"))
        print(f"{campo:22s} {m['accuracy']:9.3f} {f1:9.3f}")
    print(f"\nErrori totali: {len(errori)} (dettaglio in results/errori_nlp_gold.csv)")
