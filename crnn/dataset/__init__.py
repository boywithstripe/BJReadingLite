from ._360cc import _360CC
from ._own import _OWN
from .data_randAugment import data_randAugment

def get_dataset(config):

    if config.DATASET.DATASET == "360CC":
        return _360CC
    elif config.DATASET.DATASET == "OWN":
        return _OWN
    elif config.DATASET.DATASET == "RANDAUGMENT":
        return data_randAugment
    else:
        raise NotImplemented()

def get_val_dataset(config):

    if config.DATASET.VALDATASET == "360CC":
        return _360CC
    elif config.DATASET.VALDATASET == "OWN":
        return _OWN
    else:
        raise NotImplemented()