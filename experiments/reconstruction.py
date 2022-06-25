import os
import sys
import hydra
from omegaconf import DictConfig
import os
import sirf.STIR as pet
pet.set_verbosity(0)
pet.AcquisitionData.set_storage_scheme("memory")
pet.MessageRedirector(info=None, warn=None, errr=None)

sys.path.append(
    os.path.dirname(
        os.getcwd()
        )
    )

from src import (
    ModelClass, DatasetClass, PriorClass
    )

@hydra.main(config_path='../cfgs', config_name='config')
def reconstruction(cfg : DictConfig) -> None:
    
    dataset = DatasetClass(cfg)

    PriorClass(cfg, dataset)
    
    ModelClass(cfg, dataset)


if __name__ == '__main__':
    reconstruction()