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


# Parole chiave usate dal generatore e dall'estrattore NLP.
# Ogni valore è una lista di espressioni italiane che indicano il valore.
# Il generatore sceglie da qui; l'estrattore cerca le stesse espressioni.
# Il gold set manuale NON usa solo queste parole, per evitare un test circolare.
PAROLE_CHIAVE = {
    TipoEdificio.RESIDENZIALE: ["palazzina", "condominio", "appartamento", "abitazione"],
    TipoEdificio.INDUSTRIALE: ["capannone", "stabilimento", "impianto industriale", "deposito"],
    TipoEdificio.BOSCO: ["bosco", "sterpaglie", "pineta", "area boschiva"],
    "sostanze_pericolose_si": ["bombole", "serbatoio", "gas", "solventi", "acido", "vernici"],
    "persone_intrappolate_si": ["intrappolat", "bloccat", "non riesce a uscire", "dentro"],
    "persone_intrappolate_incerte": ["forse", "non sappiamo", "potrebbe esserci"],
}

# Soglia di altezza (in piani) sopra cui serve l'autoscala. Assunzione dichiarata.
SOGLIA_PIANI_AUTOSCALA = 4

# Soglia di probabilità per trasformare la distribuzione BN in una decisione.
# Assunzione dichiarata: P(Critico) > soglia => gravità operativa "critica".
SOGLIA_PROB_CRITICO = 0.7

# Velocità massima ammessa sulla griglia (unità arbitraria di distanza per unità di tempo).
# Usata per h(n) = distanza_euclidea / V_MAX_GRIGLIA, con V_MAX_GRIGLIA = velocità massima di qualunque arco.
V_MAX_GRIGLIA = 1.0
