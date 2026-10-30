"""Test della ricerca: ammissibilità, consistenza e ottimalità delle euristiche """

import random
import sys
from collections import deque
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from domain import TipoMezzo
import search as s  


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


def _scenario_casuale(rng, n_incidenti, n_mezzi, penalita=10):
    """Ruoli (1-3 per incidente), mezzi adatti casuali, costi in [1, 6] con archi irraggiungibili."""
    mezzi = tuple(f"m{k}" for k in range(n_mezzi))
    adatti, costo = {}, {}
    for k in range(n_incidenti):
        i = f"i{k}"
        for m in mezzi:
            costo[(m, i)] = s.INF if rng.random() < 0.1 else max(rng.randint(0, 6), s.COSTO_MINIMO)
        for ruolo in ["estinzione", "rinforzo", "soccorso_quota"][:rng.randint(1, 3)]:
            adatti[(i, ruolo)] = [m for m in mezzi if rng.random() < 0.5]
    return s.scenario_da_adatti(adatti, mezzi, costo, penalita)


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


@pytest.mark.parametrize("seme", range(30))
def test_dispatch_euristiche_ammissibili_consistenti_e_algoritmi_ottimi(seme):
    rng = random.Random(seme)
    sc = _scenario_casuale(rng, n_incidenti=rng.randint(2, 4), n_mezzi=rng.randint(2, 6))
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
            assert h(st) <= c + h(st2)

    esito = ottimo(p.inizio())
    for euristica in (lambda st: 0, p.h_min, p.h_libero):
        r = s.ricerca(p.inizio(), p.obiettivo, p.successori, euristica)
        assert r.costo == esito
        assegnazioni, scoperti = p.piano(r.cammino)
        assert len(assegnazioni) + len(scoperti) == len(sc.ruoli)
        assert len({m for m, _, _ in assegnazioni}) == len(assegnazioni)  # mezzi distinti
        costo_piano = sum(sc.costo[(m, i)] for m, i, _ in assegnazioni) + sc.penalita * len(scoperti)
        assert costo_piano == esito


def test_ruolo_scoperto_quando_manca_un_mezzo():
    # Due ruoli, un solo mezzo adatto a entrambi: uno dei due resta scoperto.
    sc = s.scenario_da_adatti({("i0", "estinzione"): ["m0"], ("i0", "rinforzo"): ["m0"]},
                              ("m0",), {("m0", "i0"): 3}, penalita=10)
    p = s.Dispatch(sc)
    r = s.ricerca(p.inizio(), p.obiettivo, p.successori, p.h_libero)
    assegnazioni, scoperti = p.piano(r.cammino)
    assert r.costo == 13 and len(assegnazioni) == 1 and len(scoperti) == 1


def test_ordine_mrv_mette_prima_i_ruoli_piu_vincolati():
    sc = s.scenario_da_adatti({("i0", "estinzione"): ["m0", "m1", "m2"], ("i1", "soccorso_quota"): ["m1"]},
                              ("m0", "m1", "m2"), {(m, i): 2 for m in ("m0", "m1", "m2") for i in ("i0", "i1")})
    assert sc.ruoli[0] == ("i1", "soccorso_quota")


def test_costo_minimo_di_una_assegnazione_e_uno():
    # Mezzo e incidente nella stessa cella: distanza 0, ma l'arco costa COSTO_MINIMO.
    g = s.Griglia(3, 3, frozenset())
    caso = s.Caso(g, {"m0": (TipoMezzo.AUTOPOMPA, (1, 1))}, {"i0": (None, (1, 1))})
    sc = s.prepara_scenario(caso, lambda c: {("i0", "estinzione"): ["m0"]})
    assert sc.costo[("m0", "i0")] == s.COSTO_MINIMO
