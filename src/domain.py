"""Vocabolario unico del dominio SO-VVF-NEXT.

Generatore, adattatore NLP, KB Prolog, rete bayesiana, ricerca e ML importano da qui,
così le etichette non possono divergere tra i moduli.
"""

from enum import Enum


class TipoEdificio(str, Enum):
    RESIDENZIALE = "residenziale"
    INDUSTRIALE = "industriale"
    BOSCO = "bosco"


class SostanzePericolose(str, Enum):
    SI = "si"
    NO = "no"


class Gravita(str, Enum):
    """Gravità a tre livelli. La regola che la definisce è in modello_generativo.regola_gravita e in kb.pl."""
    CRITICO = "critico"
    ALTO = "alto"
    BASSO = "basso"


class VelocitaFuoco(str, Enum):
    ALTA = "alta"
    MEDIA = "media"
    BASSA = "bassa"


class Persone(str, Enum):
    CONFERMATE = "confermate"
    INCERTE = "incerte"
    ASSENTI = "assenti"


class Meteo(str, Enum):
    VENTO_FORTE = "vento_forte"
    CALMO = "calmo"


class Vulnerabilita(str, Enum):
    """Vulnerabilità strutturale della zona: fatto del registro territoriale, non presente nella chiamata."""
    ALTA = "alta"
    BASSA = "bassa"


class TipoMezzo(str, Enum):
    AUTOPOMPA = "autopompa"
    AUTOSCALA = "autoscala"
    NUCLEO_NBCR = "nucleo_nbcr"      # rischio chimico, biologico, radiologico
    CANADAIR = "canadair"


class Equipaggiamento(str, Enum):
    TUTA_HAZMAT = "tuta_hazmat"
    RILEVATORE_GAS = "rilevatore_gas"
    AUTORESPIRATORE = "autorespiratore"


class Campo(str, Enum):
    """Campi estratti dal testo della chiamata."""
    TIPO_EDIFICIO = "tipo_edificio"
    SOSTANZE_PERICOLOSE = "sostanze_pericolose"
    PERSONE = "persone"


# Parole chiave usate dal generatore. Il gold set manuale non le usa tutte,
# per evitare un test circolare.
PAROLE_CHIAVE: dict[Campo, dict[Enum, list[str]]] = {
    Campo.TIPO_EDIFICIO: {
        TipoEdificio.RESIDENZIALE: ["una palazzina", "un condominio", "un appartamento"],
        TipoEdificio.INDUSTRIALE: ["un capannone", "uno stabilimento", "un deposito"],
        TipoEdificio.BOSCO: ["un bosco", "delle sterpaglie", "una pineta"],
    },
    Campo.SOSTANZE_PERICOLOSE: {
        SostanzePericolose.SI: ["bombole di gas", "un serbatoio", "delle vernici", "dei solventi", "dell'acido"],
        SostanzePericolose.NO: ["nessuna sostanza pericolosa", "niente sostanze pericolose", "nessun rischio chimico"],
    },
    Campo.PERSONE: {
        Persone.CONFERMATE: ["ci sono persone intrappolate", "c'è qualcuno bloccato", "qualcuno non riesce a uscire"],
        Persone.INCERTE: ["non sappiamo se ci sia qualcuno", "forse c'è qualcuno dentro", "potrebbe esserci qualcuno"],
        Persone.ASSENTI: ["non c'è nessuno dentro", "tutti sono usciti", "l'edificio è vuoto"],
    },
}

# Altezza in piani oltre la quale serve l'autoscala. Assunzione dichiarata.
SOGLIA_PIANI_AUTOSCALA = 4

# Soglia di probabilità per trasformare la distribuzione della rete bayesiana in decisione.
# Assunzione dichiarata: P(Critico) > soglia => gravità operativa "critica".
SOGLIA_PROB_CRITICO = 0.7

# Denominatore dell'euristica di A* spaziale: h = distanza_euclidea / V_MAX_GRIGLIA.
# Deve essere >= alla velocità di ogni arco, altrimenti h non è ammissibile.
V_MAX_GRIGLIA = 1.0
