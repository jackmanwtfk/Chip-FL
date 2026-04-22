from .CL.trainer import CLTrainer

from .FL.trainer import FedTrainer
from .FL.distributed.client_trainer import FedClientTrainer
from .FL.distributed.server_trainer import FedServerTrainer

from .SL.trainer import SplitTrainer
from .SL.distributed.client_trainer import SplitClientTrainer
from .SL.distributed.server_trainer import SplitServerTrainer

from .Ampere.trainer import AmpereTrainer
from .Ampere.distributed.client_trainer import AmpereClientTrainer
from .Ampere.distributed.server_trainer import AmpereServerTrainer

from .PreRout.trainer import PreRoutFedTrainer
from .FedEDA.trainer import FedEDATrainer


__all__ = [
    'CLTrainer',

    'FedTrainer',
    'FedClientTrainer',
    'FedServerTrainer',

    'SplitTrainer',
    'SplitClientTrainer',
    'SplitServerTrainer',

    'AmpereTrainer',
    'AmpereClientTrainer',
    'AmpereServerTrainer',

    'PreRoutFedTrainer',
    'FedEDATrainer'
]