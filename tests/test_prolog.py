"""Test della knowledge base Prolog. Richiedono SWI-Prolog e pyswip: altrimenti vengono saltati."""

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
import modello_generativo as mg  # noqa: E402
from domain import (  # noqa: E402
    Gravita,
    Meteo,
    Persone,
    SostanzePericolose,
    TipoEdificio,
    TipoMezzo,
    VelocitaFuoco,
    Vulnerabilita,
)


@pytest.fixture(scope="module")
def kb():
    pytest.importorskip("pyswip")
    from prolog_engine import KnowledgeBase
    try:
        return KnowledgeBase()
    except Exception as e:  # SWI-Prolog non installato o non trovato
        pytest.skip(f"SWI-Prolog non disponibile: {e}")


def _incidente(i, tipo=TipoEdificio.RESIDENZIALE, piano=0, sost=SostanzePericolose.NO,
               persone=Persone.ASSENTI, meteo=Meteo.CALMO):
    return dict(id=i, tipo=tipo, piano=piano, sostanze=sost, persone=persone, meteo=meteo)


def test_soglia_autoscala_coerente_con_domain(kb):
    from domain import SOGLIA_PIANI_AUTOSCALA
    kb.carica_scenario([], {}, {}, [], {})
    r = kb._q("soglia_piani_autoscala(S)")
    assert int(r[0]["S"]) == SOGLIA_PIANI_AUTOSCALA


def test_autoscala_solo_in_altezza(kb):
    inc = [_incidente("a3", TipoEdificio.RESIDENZIALE, 3), _incidente("a4", TipoEdificio.RESIDENZIALE, 4),
           _incidente("b5", TipoEdificio.INDUSTRIALE, 5)]
    mezzi = {"m_scala": TipoMezzo.AUTOSCALA, "m_pompa": TipoMezzo.AUTOPOMPA}
    kb.carica_scenario(inc, mezzi, {i["id"]: (Vulnerabilita.BASSA, 0) for i in inc}, [],
                       {i["id"]: VelocitaFuoco.BASSA for i in inc})
    assert "m_scala" not in kb.mezzi_idonei("a3")
    assert "m_scala" in kb.mezzi_idonei("a4")
    assert "m_scala" in kb.mezzi_idonei("b5")


def test_negazione_canadair_non_con_sostanze(kb):
    inc = [_incidente("bosco1", TipoEdificio.BOSCO, 0, SostanzePericolose.NO),
           _incidente("bosco2", TipoEdificio.BOSCO, 0, SostanzePericolose.SI)]
    kb.carica_scenario(inc, {"cnd": TipoMezzo.CANADAIR}, {i["id"]: (Vulnerabilita.BASSA, 0) for i in inc},
                       [], {i["id"]: VelocitaFuoco.BASSA for i in inc})
    assert kb.mezzi_idonei("bosco1") == ["cnd"]
    assert kb.mezzi_idonei("bosco2") == []


def test_gravita_coerente_con_regola_python(kb):
    """Stessa regola in Prolog e in Python, su molte situazioni casuali: devono coincidere sempre."""
    rng = random.Random(config.SEME)
    casi = [mg.genera_variabili(rng) for _ in range(2000)]
    inc, reg, vel, attesi = [], {}, {}, {}
    for n, v in enumerate(casi):
        i = f"c{n}"
        inc.append(_incidente(i, v["tipo_edificio"], v["piano"], v["sostanze_pericolose"],
                              v["persone"], v["meteo"]))
        reg[i] = (v["vulnerabilita"], v["adiacenti"])
        vel[i] = v["velocita_fuoco"]
        attesi[i] = mg.regola_gravita(v).value
    kb.carica_scenario(inc, {}, reg, [], vel)
    difformi = [i for i in attesi if kb.gravita(i) != attesi[i]]
    assert not difformi, f"gravità diverse da regola_gravita su {len(difformi)} casi, es. {difformi[:3]}"


def test_abduzione_contiene_la_velocita_vera(kb):
    """Se la gravità osservata è quella prodotta dalla velocità vera, quella velocità è tra le ipotesi."""
    rng = random.Random(config.SEME + 1)
    casi = [mg.genera_variabili(rng) for _ in range(300)]
    inc, reg, vel = [], {}, {}
    for n, v in enumerate(casi):
        i = f"h{n}"
        inc.append(_incidente(i, v["tipo_edificio"], v["piano"], v["sostanze_pericolose"],
                              v["persone"], v["meteo"]))
        reg[i] = (v["vulnerabilita"], v["adiacenti"])
        vel[i] = v["velocita_fuoco"]
    kb.carica_scenario(inc, {}, reg, [], vel)
    for n, v in enumerate(casi):
        i = f"h{n}"
        g = kb.gravita(i)
        assert v["velocita_fuoco"].value in kb.spiegazioni_velocita(i, g)


def test_propagazione_ricorsiva_e_ciclo_terminano(kb):
    """Catena a, b, c e ciclo a-b-c-a: la ricorsione deve terminare e raggiungere tutti."""
    # ca critica (persone confermate, velocita alta, vulnerabilita alta, 2 adiacenti: punteggio 6).
    # cb e cc bassa: esposte per collegamento con ca, anche attraverso il ciclo.
    inc = [_incidente("ca", TipoEdificio.INDUSTRIALE, 1, SostanzePericolose.NO, Persone.CONFERMATE),
           _incidente("cb", TipoEdificio.INDUSTRIALE, 1, SostanzePericolose.NO, Persone.ASSENTI),
           _incidente("cc", TipoEdificio.INDUSTRIALE, 1, SostanzePericolose.NO, Persone.ASSENTI)]
    reg = {"ca": (Vulnerabilita.ALTA, 2), "cb": (Vulnerabilita.BASSA, 0), "cc": (Vulnerabilita.BASSA, 0)}
    vel = {"ca": VelocitaFuoco.ALTA, "cb": VelocitaFuoco.BASSA, "cc": VelocitaFuoco.BASSA}
    kb.carica_scenario(inc, {}, reg, [("ca", "cb"), ("cb", "cc"), ("cc", "ca")], vel)
    assert kb.gravita("ca") == "critico"
    assert kb.collegati("ca") == {"cb", "cc"}
    assert kb.esposto("cb") is True
    assert kb.esposto("cc") is True
    assert kb.esposto("ca") is False         # un incidente non è esposto a se stesso


def test_gravita_operativa_eleva_alto_esposto_a_critico(kb):
    # i1 critica (persone confermate, velocita alta, sostanze si, vulnerabilita alta, 2 adiacenti),
    # i2 alta (assenti, velocita alta, sostanze si) ma adiacente a i1
    inc = [_incidente("i1", TipoEdificio.INDUSTRIALE, 2, SostanzePericolose.SI, Persone.CONFERMATE, Meteo.VENTO_FORTE),
           _incidente("i2", TipoEdificio.INDUSTRIALE, 0, SostanzePericolose.SI, Persone.ASSENTI, Meteo.VENTO_FORTE)]
    reg = {"i1": (Vulnerabilita.ALTA, 2), "i2": (Vulnerabilita.BASSA, 0)}
    vel = {"i1": VelocitaFuoco.ALTA, "i2": VelocitaFuoco.ALTA}
    kb.carica_scenario(inc, {}, reg, [("i1", "i2")], vel)
    assert kb.gravita("i2") == "alto"
    assert kb.gravita_operativa("i2") == "critico"
    assert kb.gravita("i1") == "critico"


def test_violazioni_proposta_valida_e_non(kb):
    inc = [_incidente("v1", TipoEdificio.RESIDENZIALE, 4), _incidente("v2", TipoEdificio.BOSCO, 0)]
    mezzi = {"m_scala": TipoMezzo.AUTOSCALA, "m_cnd": TipoMezzo.CANADAIR}
    kb.carica_scenario(inc, mezzi, {i["id"]: (Vulnerabilita.BASSA, 0) for i in inc}, [],
                       {i["id"]: VelocitaFuoco.BASSA for i in inc})
    assert kb.violazioni([("m_scala", "v1"), ("m_cnd", "v2")]) == []
    # m_scala a due incidenti: esclusività violata; m_cnd al residenziale: non idoneo
    viol = kb.violazioni([("m_scala", "v1"), ("m_scala", "v2"), ("m_cnd", "v1")])
    assert any(v.startswith("doppia_assegnazione(m_scala)") for v in viol)
    assert any(v.startswith("non_idoneo(m_cnd,v1)") for v in viol)
    # solo v1 coperto: v2 risulta scoperto
    assert "incidente_scoperto(v2)" in kb.violazioni([("m_scala", "v1")])


def test_complessita_kb_riportata(kb):
    c = kb.complessita()
    assert c["predicati"] > 0 and c["regole"] > 0 and c["clausole"] >= c["regole"]
