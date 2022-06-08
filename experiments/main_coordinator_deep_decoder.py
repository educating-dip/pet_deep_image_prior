import os
import sys
import hydra
from omegaconf import DictConfig
import os
import sirf.STIR as pet
import numpy as np

sys.path.append(
    os.path.dirname(
        os.getcwd()
        )
    )

from src import (ComputeImageMetrics, DeepDecoderPriorReconstructor,
    ObjectiveFunctionModule 
    )

pet.set_verbosity(0)

@hydra.main(config_path='../cfgs', config_name='config_deep_decoder')
def baselines(cfg : DictConfig) -> None:
    
    # GET THE DATA
    prompts = pet.AcquisitionData(cfg.dataset.prompts)
    prompts.set_storage_scheme("memory")
    additive_factors = pet.AcquisitionData(cfg.dataset.additive)
    multiplicative_factors = pet.AcquisitionData(cfg.dataset.multiplicative)
    
    # GET RECONSTRUCTION "VOLUME"
    image = prompts.create_uniform_image(1.0).zoom_image(
            zooms=(1., 1., 1.),
            offsets_in_mm=(0., 0., 0.),
            size=(-1, cfg.dataset.image_xy, cfg.dataset.image_xy)
        )

    # SET UP THE SENSITIVITY MODEL
    normalisation_model = pet.AcquisitionSensitivityModel(multiplicative_factors)
    normalisation_model.set_up(prompts)
    sensitivity_vals = prompts.get_uniform_copy(1.0)
    normalisation_model.normalise(sensitivity_vals)
    sensitivity_factors = pet.AcquisitionSensitivityModel(sensitivity_vals)
    sensitivity_factors.set_up(prompts)

    # SET UP THE ACQUISITION MODEL
    acquisition_model = pet.AcquisitionModelUsingParallelproj()
    acquisition_model.set_up(prompts,image)
    acquisition_model.set_additive_term(additive_factors)
    acquisition_model.set_acquisition_sensitivity(sensitivity_factors)

    
    # SET UP THE OBJECTIVE FUNCTIONAL
    objective_functional = pet.make_Poisson_loglikelihood(prompts, acq_model=acquisition_model)
    objective_functional.set_recompute_sensitivity(1)
    initial = image.clone()

    # SET THE PRIOR
    if cfg.prior.name == "OSEM":
        initial = image
        print("Just plain old OSEM")
        objective_functional.set_up(image)
    elif cfg.prior.name == "QP":
        prior = pet.QuadraticPrior()
        print('using Quadratic prior...')
        prior.set_penalisation_factor(cfg.prior.penalty_factor)
        if cfg.prior.initial == True:
            initial = image.clone().fill(pet.ImageData(cfg.dataset.initial))
        if cfg.prior.kappa == True:
            kappa = pet.ImageData(cfg.dataset.kappa)
            prior.set_kappa(image.clone().fill(kappa))
        prior.set_up(image)
        objective_functional.set_prior(prior)
        objective_functional.set_up(image)
    elif cfg.prior.name == "RDP":
        prior = pet.RelativeDifferencePrior()
        print('using Relative Difference prior...')
        prior.set_penalisation_factor(cfg.prior.penalty_factor)
        prior.set_gamma(cfg.prior.gamma)
        if cfg.prior.initial == True:
            initial = image.clone().fill(pet.ImageData(cfg.dataset.initial))
        if cfg.prior.kappa == True:
            kappa = pet.ImageData(cfg.dataset.kappa)
            prior.set_kappa(image.clone().fill(kappa))
        prior.set_up(image)
        objective_functional.set_prior(prior)
        objective_functional.set_up(image)
    else:
        raise NotImplementedError


    # SETUP THE QUALITY METRICS
    if cfg.dataset.name == "2D":

        ROIs = ["ROI_LungLesion"]
        ROIs_masks = []
        ROIs_b_mask = np.load(
            cfg.dataset.quality_path + "/" + "ROI_Lung" + ".npy"
        )

        for i in range(len(ROIs)):
            ROIs_masks.append(np.load(
                cfg.dataset.quality_path + "/" + ROIs[i] + ".npy")
                )
        
        # LungLesion 3254.626, Lung 1254.6259
        emissions = [3254.626, 1254.6259]
        
    elif cfg.dataset.name == "3D":

        ROIs = ["ROI_AbdominalWallLesion","ROI_HeartLesion","ROI_LiverLesion","ROI_LungLesion","ROI_SpineLesion"]
        ROIs_masks = []
        ROIs_b_mask = np.load(cfg.dataset.quality_path + "/" + "ROI_Liver" + ".npy")
        for i in range(len(ROIs)):
            ROIs_masks.append(np.load(
                cfg.dataset.quality_path + "/" + ROIs[i] + ".npy")
            )
        emissions = [2897.9812,3254.626,1254.6259,0]
        
    image_metrics = ComputeImageMetrics(
        emissions=emissions,
        ROIs_a=ROIs_masks,
        ROIs_b=ROIs_b_mask
    )
    

    reconstructor = DeepDecoderPriorReconstructor(
            obj_fun_module = ObjectiveFunctionModule(
                image_template=image.clone(), 
                obj_fun = objective_functional
                ), 
        image_template=image.clone(),
        cfgs=cfg
    )
    reconstructor.reconstruct(
        image_metrics
    )

if __name__ == '__main__':
    baselines()