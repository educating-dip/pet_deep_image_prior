import os
import hydra
from omegaconf import DictConfig
import os
import sirf.STIR as pet
import sirf.STIR
import datetime 
import socket
import numpy
import tensorboardX
import numpy as np

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
    if inplace:
        x -= x.min()
        x /= x.max()
    else:
        x = x - x.min()
        x = x / x.max()
    return x



@hydra.main(config_path='cfgs', config_name='config')
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

    # SET THE PRIOR
    print(cfg.prior.name)
    if cfg.prior.name == "osem":
        initial = image
        print("Just plain old OSEM")
    elif cfg.prior.name == "qp":
        prior = pet.QuadraticPrior()
        print('using Quadratic prior...')
        prior.set_penalisation_factor(cfg.prior.penalty_factor)
        if cfg.prior.initial == True:
            initial = pet.ImageData(cfg.dataset.initial)
        if cfg.prior.kappa == True:
            kappa = pet.ImageData(cfg.dataset.kappa)
            prior.set_kappa(image.fill(kappa))
            objective_functional.set_prior(prior)
        prior.set_up(image)
    elif cfg.prior.name == "rdp":
        prior = pet.RelativeDifferencePrior()
        print('using Relative Difference prior...')
        prior.set_penalisation_factor(cfg.prior.penalty_factor)
        prior.set_gamma(cfg.prior.gamma)
        if cfg.prior.initial == True:
            initial = pet.ImageData(cfg.dataset.initial)
        if cfg.prior.kappa == True:
            kappa = pet.ImageData(cfg.dataset.kappa)
            prior.set_kappa(image.fill(kappa))
            objective_functional.set_prior(prior)
        prior.set_up(image)
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
    #sirf_reconstruction.process()


    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    comment = cfg.prior.name
    logdir = os.path.join(
        './',
        current_time + '_' + socket.gethostname() + comment)
    writer = tensorboardX.SummaryWriter(logdir=logdir)

    current_image = initial
    for i in range(0, cfg.dataset.num_subsets*cfg.dataset.num_epochs + 1):
        sirf_reconstruction.update(current_image)
        writer.add_image('recon', normalize(current_image.as_array()), i)
    
        """ if ground_truth is not None:
            output_psnr = PSNR(curr_image.as_array()[0], ground_truth.as_array()[0])
            writer.add_scalar('output_psnr', output_psnr, i) """

    writer.close()

if __name__ == '__main__':
    baselines()