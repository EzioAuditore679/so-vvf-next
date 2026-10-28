"""Vocabolario unico del dominio SO-VVF-NEXT.

Fonte di verità per generatore di dataset, estrattore NLP, ontologia OWL,
knowledge base Prolog, rete bayesiana e CSP. Ogni modulo importa da qui,
così le etichette non possono divergere tra un componente e l'altro.
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
    CRITICO = "critico"
    ALTO = "alto"
    MEDIO = "medio"
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
        TipoEdificio.RESIDENZIALE: ["palazzina", "condominio", "appartamento", "abitazione"],
        TipoEdificio.INDUSTRIALE: ["capannone", "stabilimento", "impianto industriale", "deposito"],
        TipoEdificio.BOSCO: ["bosco", "sterpaglie", "pineta", "area boschiva"],
    },
    Campo.SOSTANZE_PERICOLOSE: {
        SostanzePericolose.SI: ["bombole", "serbatoio", "gas", "solventi", "acido", "vernici"],
        SostanzePericolose.NO: ["nessuna sostanza", "niente sostanze", "nessun rischio chimico"],
    },
    Campo.PERSONE: {
        Persone.CONFERMATE: ["intrappolat", "bloccat", "non riesce a uscire"],
        Persone.INCERTE: ["forse", "non sappiamo", "potrebbe esserci"],
        Persone.ASSENTI: ["nessuno dentro", "tutti usciti", "nessuno ferito"],
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
