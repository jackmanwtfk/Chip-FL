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

try:
    from .FedEDA.trainer import FedEDATrainer
except ModuleNotFoundError:
    FedEDATrainer = None


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
]

if FedEDATrainer is not None:
    __all__.append('FedEDATrainer')
