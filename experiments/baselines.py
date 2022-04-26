import os
import sys
import hydra
from omegaconf import DictConfig
import os
import sirf.STIR as pet
import datetime 
import socket
import tensorboardX
import numpy as np

sys.path.append(
    os.path.dirname(
        os.getcwd()
        )
    )

from src import ComputeImageMetrics, normalize


pet.set_verbosity(0)

@hydra.main(config_path='../cfgs', config_name='config_baselines')
def baselines(cfg : DictConfig) -> None:
    
    # GET THE DATA
    prompts = pet.AcquisitionData(cfg.dataset.prompts)
    prompts.set_storage_scheme("memory")
    additive_factors = pet.AcquisitionData(cfg.dataset.additive)
    multiplicative_factors = pet.AcquisitionData(cfg.dataset.multiplicative)
    
    # GET RECONSTRUCTION "VOLUME"
    image = prompts.create_uniform_image(1.0,cfg.dataset.image_xy)

    # SET UP THE SENSITIVITY MODEL
    normalisation_model = pet.AcquisitionSensitivityModel(multiplicative_factors)
    normalisation_model.set_up(prompts)
    sensitivity_vals = prompts.get_uniform_copy(1.0)
    normalisation_model.normalise(sensitivity_vals)
    sensitivity_factors = pet.AcquisitionSensitivityModel(sensitivity_vals)
    sensitivity_factors.set_up(prompts)

    # SET UP THE ACQUISITION MODEL
    acquisition_model = pet.AcquisitionModelUsingRayTracingMatrix()
    acquisition_model.set_num_tangential_LORs(cfg.dataset.num_LORs)
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
    else:
        raise NotImplementedError

    # CREATE RECONSTRUCTION OBJECT
    sirf_reconstruction = pet.OSMAPOSLReconstructor()
    sirf_reconstruction.set_objective_function(objective_functional)
    num_subsets = cfg.dataset.num_subsets
    num_subiterations = cfg.dataset.num_subsets*cfg.dataset.num_epochs
    sirf_reconstruction.set_num_subsets(num_subsets)
    sirf_reconstruction.set_num_subiterations(num_subiterations)

    # INITIALISE THE RECONSTRUCTION OBJECT
    sirf_reconstruction.set_up(image)
    sirf_reconstruction.set_current_estimate(initial)

    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    logdir = os.path.join('./', current_time + '_' + socket.gethostname())
    writer = tensorboardX.SummaryWriter(logdir=logdir)

    # SETUP THE QUALITY METRICS
    if cfg.dataset.name == "2D":

        ROIs = ["ROI_LungLesion"] # ["ROI_Heart"] #
        ROIs_masks = []
        ROIs_b_mask = np.load(
            cfg.dataset.quality_path + "/" + "ROI_Lung" + ".npy"
        )

        for i in range(len(ROIs)):
            ROIs_masks.append(np.load(
                cfg.dataset.quality_path + "/" + ROIs[i] + ".npy")
                )
        
        # Heart 2897.9812, LungLeison 3254.626, Lung 1254.6259
        emissions = [3254.626, 1254.6259]
        image_metrics = ComputeImageMetrics(
            emissions=emissions,
            ROIs_a=ROIs_masks,
            ROIs_b=ROIs_b_mask
        )
        
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

    # TO DO GET THE QUALITY METRICS
    current_image = initial
    for i in range(0, cfg.dataset.num_subsets*cfg.dataset.num_epochs + 1):
        sirf_reconstruction.update(
            current_image
            )
        (crc, std) = image_metrics.get_all_metrics(
            current_image.as_array()
            )

        writer.add_image('recon', normalize(current_image.as_array()), i)
        writer.add_scalar('CRC', crc, i)
        writer.add_scalar('STDEV', std, i)

    writer.close()

if __name__ == '__main__':
    baselines()