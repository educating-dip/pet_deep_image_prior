import os
import hydra
from omegaconf import DictConfig
import os
import sirf.STIR as pet
import datetime 
import socket
import numpy
import tensorboardX
import numpy as np
pet.set_verbosity(0)

def PSNR(reconstruction, ground_truth, data_range=None):
    gt = np.asarray(ground_truth)
    mse = np.mean((np.asarray(reconstruction) - gt)**2)
    if mse == 0.:
        return float('inf')
    if data_range is not None:
        return 20*np.log10(data_range) - 10*np.log10(mse)
    else:
        data_range = np.max(gt) - np.min(gt)
        return 20*np.log10(data_range) - 10*np.log10(mse)

def normalize(x, inplace=False):
    # Exploding pixel at edge of FOV we need to ignore...
    mask = numpy.zeros_like(x)
    mask[:,50:201, 50:201] = 1
    x = mask*x
    if inplace:
        x -= x.min()
        x /= x.max()
    else:
        x = x - x.min()
        x = x / x.max()
    return x

def CRC(ROIs_b, ROIs_a, recon, emissions):
    # CONTRAST RECOVERY COEFFICIENT
    # abar = ROI average uptake
    # Ka = number of ROIs
    # bbar = background average uptake
    # Kb = number of background ROIs
    # CRC = 1/R \sum_{r=1}^{R} (abar/bbar - 1)/(atrue/btrue - 1)
    CRCval = 0
    for i in range(len(ROIs_a)):
        abar = np.mean(recon[np.nonzero(ROIs_a[i])])
        bbar = np.mean(recon[np.nonzero(ROIs_b)])
        atrue = emissions[i]
        btrue = emissions[-1]
        CRCval += (abar/bbar - 1) / (atrue/btrue - 1)
    return CRCval/len(ROIs_a)

def STD(ROIs_b, recon):
    # STANDARD DEVIATION
    # abar = ROI average uptake
    # Ka = number of ROIs
    # bbar = background average uptake
    # Kb = number of background ROIs
    # CRC = 1/R \sum_{r=1}^{R} (abar/bbar - 1)/(atrue/btrue - 1)
    return np.std(recon[np.nonzero(ROIs_b)])


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
        ROIs_b_mask = np.load(cfg.dataset.quality_path + "/" + "ROI_Lung" + ".npy")
        for i in range(len(ROIs)):
            ROIs_masks.append(np.load(cfg.dataset.quality_path + "/" + ROIs[i] + ".npy"))
        # Heart 2897.9812, LungLeison 3254.626, Lung 1254.6259
        emissions = [3254.626,1254.6259]
    elif cfg.dataset.name == "3D":
        ROIs = ["ROI_AbdominalWallLesion","ROI_HeartLesion","ROI_LiverLesion","ROI_LungLesion","ROI_SpineLesion"]
        ROIs_masks = []
        ROIs_b_mask = np.load(cfg.dataset.quality_path + "/" + "ROI_Liver" + ".npy")
        for i in range(len(ROIs)):
            ROIs_masks.append(np.load(cfg.dataset.quality_path + "/" + ROIs[i] + ".npy"))
        emissions = [2897.9812,3254.626,1254.6259,0]

    # TO DO GET THE QUALITY METRICS
    current_image = initial
    for i in range(0, cfg.dataset.num_subsets*cfg.dataset.num_epochs + 1):
        sirf_reconstruction.update(current_image)
        writer.add_image('recon', normalize(current_image.as_array()), i)
        writer.add_scalar("CRC",CRC(ROIs_b_mask, ROIs_masks,current_image.as_array(), emissions),i)
        writer.add_scalar("STDEV", STD(ROIs_b_mask,current_image.as_array()),i)

    writer.close()

if __name__ == '__main__':
    baselines()