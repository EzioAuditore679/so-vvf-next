"""Test della ricerca: ammissibilità, consistenza e ottimalità delle euristiche (docs/ricerca.md)."""

import random
import sys
from collections import deque
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from domain import TipoMezzo  # noqa: E402
import search as s  # noqa: E402


def _griglia_casuale(rng, righe=8, colonne=8, p=0.25):
    g = s.genera_griglia(rng, righe, colonne, p)
    liberi = [(r, k) for r in range(righe) for k in range(colonne) if (r, k) not in g.ostacoli]
    return g, liberi


def _distanze_esatte(g, arrivo):
    """Distanze reali da ogni cella verso arrivo (BFS sulla griglia)."""
    dist = {arrivo: 0}
    coda = deque([arrivo])
    while coda:
        c = coda.popleft()
        for v, _ in g.successori(c):
            if v not in dist:
                dist[v] = dist[c] + 1
                coda.append(v)
    return dist


def test_manhattan_ammissibile_e_monotona_sulla_griglia():
    rng = random.Random(1)
    for _ in range(20):
        g, liberi = _griglia_casuale(rng)
        arrivo = rng.choice(liberi)
        esatte = _distanze_esatte(g, arrivo)
        for c in esatte:
            assert s.manhattan(c, arrivo) <= esatte[c]
        for c in esatte:
            for v, costo in g.successori(c):
                assert s.manhattan(c, arrivo) <= costo + s.manhattan(v, arrivo)


def test_astar_su_griglia_uguale_a_lcfs():
    rng = random.Random(2)
    for _ in range(30):
        g, liberi = _griglia_casuale(rng)
        a, b = rng.choice(liberi), rng.choice(liberi)
        r_a = s.distanza_griglia(g, a, b, con_euristica=True)
        r_l = s.distanza_griglia(g, a, b, con_euristica=False)
        assert r_a.costo == r_l.costo


def test_griglia_irraggiungibile_restituisce_inf():
    g = s.Griglia(3, 3, frozenset({(0, 1), (1, 1), (2, 1)}))
    r = s.distanza_griglia(g, (0, 0), (0, 2))
    assert r.costo == s.INF and r.completo


def _scenario_casuale(rng, n_incidenti, n_mezzi):
    incidenti = tuple(f"i{k}" for k in range(n_incidenti))
    mezzi = tuple(f"m{k}" for k in range(n_mezzi))
    idonei, costo = {}, {}
    for i in incidenti:
        scelti = [m for m in mezzi if rng.random() < 0.7] or [rng.choice(mezzi)]
        idonei[i] = scelti
        for m in scelti:
            c = rng.randint(0, 6)
            costo[(m, i)] = s.INF if rng.random() < 0.05 else max(c, s.COSTO_MINIMO)
    return s.Scenario(incidenti, mezzi, idonei, costo)


def _valore_ottimo(p: s.Dispatch):
    """Costo minimo esatto da ogni stato verso l'obiettivo (programmazione dinamica)."""
    memo = {}

    def v(st):
        if st in memo:
            return memo[st]
        if p.obiettivo(st):
            memo[st] = 0
            return 0
        best = s.INF
        for st2, c in p.successori(st):
            best = min(best, c + v(st2))
        memo[st] = best
        return best

    return v


def _stati_raggiungibili(p: s.Dispatch):
    visti = {p.inizio()}
    coda = deque([p.inizio()])
    archi = []
    while coda:
        st = coda.popleft()
        for st2, c in p.successori(st):
            archi.append((st, c, st2))
            if st2 not in visti:
                visti.add(st2)
                coda.append(st2)
    return visti, archi


@pytest.mark.parametrize("seme", range(25))
def test_dispatch_euristiche_ammissibili_consistenti_e_algoritmi_ottimi(seme):
    rng = random.Random(seme)
    sc = _scenario_casuale(rng, n_incidenti=rng.randint(2, 4), n_mezzi=rng.randint(2, 5))
    p = s.Dispatch(sc)
    ottimo = _valore_ottimo(p)
    stati, archi = _stati_raggiungibili(p)

    for st in stati:
        v = ottimo(st)
        assert p.h_min(st) <= v
        assert p.h_libero(st) <= v
        assert p.h_min(st) <= p.h_libero(st)

    for st, c, st2 in archi:
        for h in (p.h_min, p.h_libero):
            if h(st) < s.INF and h(st2) < s.INF:
                assert h(st) <= c + h(st2)

    esito = ottimo(p.inizio())
    for euristica in (lambda st: 0, p.h_min, p.h_libero):
        r = s.ricerca(p.inizio(), p.obiettivo, p.successori, euristica)
        assert r.costo == esito


def test_dispatch_senza_soluzione():
    sc = s.Scenario(("i0", "i1"), ("m0",), {"i0": ["m0"], "i1": ["m0"]},
                    {("m0", "i0"): 3, ("m0", "i1"): 4})
    p = s.Dispatch(sc)
    r = s.ricerca(p.inizio(), p.obiettivo, p.successori, p.h_libero)
    assert r.costo == s.INF and r.completo


def test_costo_minimo_di_una_assegnazione_e_uno():
    # Mezzo e incidente nella stessa cella: distanza 0, ma l'arco costa COSTO_MINIMO.
    g = s.Griglia(3, 3, frozenset())
    caso = s.Caso(g, {"m0": (TipoMezzo.AUTOPOMPA, (1, 1))}, {"i0": (None, (1, 1))})
    sc = s.prepara_scenario(caso, lambda c: {"i0": ["m0"]})
    assert sc.costo[("m0", "i0")] == s.COSTO_MINIMO
