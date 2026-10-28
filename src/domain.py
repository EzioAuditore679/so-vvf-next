"""Vocabolario unico del dominio SO-VVF-NEXT.

Fonte di verità per generatore di dataset, adattatore NLP, knowledge base
Prolog, rete bayesiana, ricerca nello spazio degli stati e ML. Ogni modulo
importa da qui, così le etichette non possono divergere tra un componente e l'altro.

Non c'è ontologia OWL né reasoner: la rappresentazione è relazionale (Prolog),
come da decisione del 28/10.
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
    """Etichetta di gravità a tre livelli. Le regole che la definiscono vanno dichiarate
    in knowledge_base/kb.pl: la definizione è una regola di dominio esplicita, non un'etichetta libera."""
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


class TipoMezzo(str, Enum):
    AUTOPOMPA = "autopompa"          # APS / autopompa serbatoio
    AUTOSCALA = "autoscala"
    NUCLEO_NBCR = "nucleo_nbcr"      # rischio chimico, biologico, radiologico
    CANADAIR = "canadair"            # incendi boschivi


class Equipaggiamento(str, Enum):
    TUTA_HAZMAT = "tuta_hazmat"
    RILEVATORE_GAS = "rilevatore_gas"
    AUTORESPIRATORE = "autorespiratore"


class Campo(str, Enum):
    """Campi estratti dal testo della chiamata. Ogni campo ha un proprio enum di valori."""
    TIPO_EDIFICIO = "tipo_edificio"
    SOSTANZE_PERICOLOSE = "sostanze_pericolose"
    PERSONE = "persone"


# Parole chiave usate dal generatore e dall'estrattore NLP.
# Struttura: Campo -> valore del campo -> lista di espressioni italiane.
# Tutte le chiavi interne sono membri degli enum sopra, mai stringhe libere.
# Il gold set manuale NON usa solo queste parole, per evitare un test circolare.
PAROLE_CHIAVE: dict[Campo, dict[Enum, list[str]]] = {
    Campo.TIPO_EDIFICIO: {
        TipoEdificio.RESIDENZIALE: ["una palazzina", "un condominio", "un appartamento"],
        TipoEdificio.INDUSTRIALE: ["un capannone", "uno stabilimento", "un deposito"],
        TipoEdificio.BOSCO: ["un bosco", "delle sterpaglie", "una pineta"],
    },
    # Le espressioni sono frasi complete, così il generatore le inserisce senza errori grammaticali.
    # L'estrattore NLP cercherà le stesse radici, non le frasi intere.
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

# Soglia di altezza (in piani) sopra cui serve l'autoscala. Assunzione dichiarata.
SOGLIA_PIANI_AUTOSCALA = 4

# Soglia di probabilità per trasformare la distribuzione BN in una decisione.
# Assunzione dichiarata: P(Critico) > soglia => gravità operativa "critica".
SOGLIA_PROB_CRITICO = 0.7

# Velocità massima ammessa sulla griglia (unità arbitraria di distanza per unità di tempo).
# Usata per h(n) = distanza_euclidea / V_MAX_GRIGLIA, con V_MAX_GRIGLIA = velocità massima di qualunque arco.
V_MAX_GRIGLIA = 1.0
