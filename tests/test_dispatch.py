"""Test dell'assegnazione dei mezzi (CSP con branch and bound): ottimo verificato con l'algoritmo ungherese."""

import itertools
import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import dispatch as d


def scenario_casuale(rng, n_incidenti, n_mezzi, penalita=10):
    """Ruoli (1-3 per incidente), mezzi adatti casuali, costi in [1, 6] con archi irraggiungibili."""
    mezzi = tuple(f"m{k}" for k in range(n_mezzi))
    adatti, costo = {}, {}
    for k in range(n_incidenti):
        i = f"i{k}"
        for m in mezzi:
            costo[(m, i)] = d.INF if rng.random() < 0.1 else rng.randint(1, 6)
        for ruolo in ["estinzione", "rinforzo", "soccorso_quota"][:rng.randint(1, 3)]:
            adatti[(i, ruolo)] = [m for m in mezzi if rng.random() < 0.5]
    return d.scenario_da_adatti(adatti, mezzi, costo, penalita)


def ottimo_ungherese(sc, usati=frozenset(), ruoli=None) -> float:
    """Costo minimo esatto: assegnamento ruoli-mezzi, con una colonna "scoperto" per ogni ruolo."""
    ruoli = list(sc.ruoli if ruoli is None else ruoli)
    if not ruoli:
        return 0
    mezzi = [m for m in sc.mezzi if m not in usati]
    grande = 1e9
    c = np.full((len(ruoli), len(mezzi) + len(ruoli)), grande)
    for a, r in enumerate(ruoli):
        for b, m in enumerate(mezzi):
            if m in sc.adatti.get(r, []) and sc.costo.get((m, r[0]), d.INF) < d.INF:
                c[a, b] = sc.costo[(m, r[0])]
        c[a, len(mezzi) + a] = sc.penalita
    righe, colonne = linear_sum_assignment(c)
    return float(c[righe, colonne].sum())


@pytest.mark.parametrize("seme", range(40))
def test_branch_and_bound_trova_l_ottimo(seme):
    rng = random.Random(seme)
    sc = scenario_casuale(rng, rng.randint(2, 5), rng.randint(2, 8))
    atteso = ottimo_ungherese(sc)
    for usa_stima in (True, False):
        s = d.risolvi(sc, usa_stima=usa_stima)
        assert s.completa and s.costo == atteso
        # Piano valido: mezzi distinti e adatti, ogni ruolo deciso una volta, costo coerente.
        mezzi = [m for m, _, _ in s.assegnazioni]
        assert len(mezzi) == len(set(mezzi))
        assert all(m in sc.adatti[(i, r)] for m, i, r in s.assegnazioni)
        decisi = [(i, r) for _, i, r in s.assegnazioni] + s.scoperti
        assert sorted(decisi) == sorted(sc.ruoli)
        assert sum(sc.costo[(m, i)] for m, i, _ in s.assegnazioni) + sc.penalita * len(s.scoperti) == s.costo


@pytest.mark.parametrize("seme", range(20))
def test_stima_ammissibile_in_ogni_stato(seme):
    """La stima non supera mai il costo minimo dei ruoli residui, per ogni scelta di ruoli e mezzi già usati."""
    rng = random.Random(1000 + seme)
    sc = scenario_casuale(rng, rng.randint(2, 4), rng.randint(2, 6))
    dominio = d.domini(sc)
    for _ in range(30):
        residui = [r for r in sc.ruoli if rng.random() < 0.7]
        usati = frozenset(m for m in sc.mezzi if rng.random() < 0.3)
        assert d.stima(sc, dominio, residui, usati) <= ottimo_ungherese(sc, usati, residui) + 1e-9


def test_correzione_di_capacita():
    # Tre ruoli, un solo mezzo adatto a tutti: almeno due restano scoperti, e la stima lo sa già alla radice.
    adatti = {("i0", "estinzione"): ["m0"], ("i1", "estinzione"): ["m0"], ("i2", "estinzione"): ["m0"]}
    sc = d.scenario_da_adatti(adatti, ("m0",), {("m0", "i0"): 1, ("m0", "i1"): 2, ("m0", "i2"): 3}, penalita=10)
    assert d.stima(sc, d.domini(sc), list(sc.ruoli), frozenset()) == 1 + 10 + 10
    s = d.risolvi(sc)
    assert s.costo == 21 and s.assegnazioni == [("m0", "i0", "estinzione")] and len(s.scoperti) == 2


def test_ruolo_senza_mezzi_resta_scoperto():
    sc = d.scenario_da_adatti({("i0", "soccorso_quota"): []}, ("m0",), {("m0", "i0"): 3}, penalita=10)
    s = d.risolvi(sc)
    assert s.costo == 10 and s.scoperti == [("i0", "soccorso_quota")]
