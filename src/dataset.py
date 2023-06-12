import sirf.STIR as pet
import numpy as np
from .deep_image_prior.utils import ComputeImageMetrics

class DatasetClass:
    def __init__(self,cfg):
        if cfg.dataset.name == '2D':
            objective_function, acquisition_model, prompts, image, sensitivity_image = Dataset2D(
                                            cfg.dataset.prompts,
                                            cfg.dataset.additive,
                                            cfg.dataset.multiplicative,
                                            cfg.dataset.image_xy)
            self.sensitivity_image = sensitivity_image

        elif cfg.dataset.name == '3D_high':
            objective_function, acquisition_model, prompts, image, sensitivity_image = Dataset3D(
                                            cfg.dataset.prompts,
                                            cfg.dataset.additive,
                                            cfg.dataset.multiplicative,
                                            cfg.dataset.image_xy,
                                            cfg.dataset.voxel_size)
            self.sensitivity_image = sensitivity_image

        elif cfg.dataset.name == '3D_medium':
            objective_function, acquisition_model, prompts, image, sensitivity_image = Dataset3D(
                                            cfg.dataset.prompts,
                                            cfg.dataset.additive,
                                            cfg.dataset.multiplicative,
                                            cfg.dataset.image_xy,
                                            cfg.dataset.voxel_size)
            self.sensitivity_image = sensitivity_image

        elif cfg.dataset.name == '3D_low':
            objective_function, acquisition_model, prompts, image, sensitivity_image = Dataset3D(
                                            cfg.dataset.prompts,
                                            cfg.dataset.additive,
                                            cfg.dataset.multiplicative,
                                            cfg.dataset.image_xy,
                                            cfg.dataset.voxel_size)
            self.sensitivity_image = sensitivity_image
        else:
            raise NotImplementedError
        
        if 'kappa' in cfg.prior.keys():
            self.kappa =    self.get_kappa(
                                image,
                                cfg.prior.kappa,
                                cfg.dataset.kappa)

        if 'initial' in cfg.prior.keys():
            self.initial =  self.get_initial(
                                image,
                                cfg.prior.initial,
                                cfg.dataset.initial)
        
        self.objective_function = objective_function
        self.acquisition_model = acquisition_model
        self.prompts = prompts

        self.quality_metrics =  QualityMetrics(
                                    cfg.dataset.quality_path,
                                    cfg.dataset.ROIs_a.names,
                                    cfg.dataset.ROIs_a.emissions,
                                    cfg.dataset.ROIs_b.names,
                                    cfg.dataset.ROIs_b.emissions)

    def get_kappa(
            self, 
            image, 
            kappa_use, 
            kappa_path):

        if kappa_use:
            return image.clone().fill(pet.ImageData(kappa_path))
        else:
            return image.clone().fill(1)

    def get_initial(
            self, 
            image, 
            initial_use, 
            initial_path):

        if initial_use:
            return image.clone().fill(pet.ImageData(initial_path))
        else:
            return image.clone().fill(1)

def Dataset2D(prompts, 
        additive, 
        multiplicative, 
        image_xy):
        
    # GET THE DATA
    prompts = pet.AcquisitionData(prompts)
    additive_factors = pet.AcquisitionData(additive)
    multiplicative_factors = pet.AcquisitionData(multiplicative)
    
    # GET RECONSTRUCTION "VOLUME"
    image = prompts.create_uniform_image(1.0).zoom_image(
        zooms=(1., 1., 1.),
        offsets_in_mm=(0., 0., 0.),
        size=(-1, image_xy, image_xy)
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
    objective_function = pet.make_Poisson_loglikelihood(prompts, acq_model=acquisition_model)
    objective_function.set_recompute_sensitivity(1)
    sensitivity_image = acquisition_model.backward(prompts.get_uniform_copy(1))
    return objective_function, acquisition_model, prompts, image, sensitivity_image


def Dataset3D(prompts, 
        additive, 
        multiplicative, 
        image_xy,
        voxel_size):
        
    # GET THE DATA
    prompts = pet.AcquisitionData(prompts)
    additive_factors = pet.AcquisitionData(additive)
    multiplicative_factors = pet.AcquisitionData(multiplicative)
    
    # GET RECONSTRUCTION "VOLUME"
    image = prompts.create_uniform_image(1.0).zoom_image(
        zooms=(1., 2.1306/voxel_size, 2.1306/voxel_size),
        offsets_in_mm=(0., 0., 0.),
        size=(-1, image_xy, image_xy)
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
    acquisition_model.set_acquisition_sensitivity(sensitivity_factors)
    acquisition_model.set_additive_term(additive_factors)

    
    # SET UP THE OBJECTIVE FUNCTIONAL
    objective_function = pet.make_Poisson_loglikelihood(prompts, acq_model=acquisition_model)
    objective_function.set_recompute_sensitivity(1)
    sensitivity_image = acquisition_model.backward(prompts.get_uniform_copy(1))

    return objective_function, acquisition_model, prompts, image, sensitivity_image
    
def QualityMetrics(
        quality_path,
        ROIs_a_names,
        ROIs_a_emissions,
        ROIs_b_names,
        ROIs_b_emissions):

    ROIs_a_masks = []
    ROIs_b_masks = []

    for i in range(len(ROIs_a_names)):
        ROIs_a_masks.append(np.load(
            quality_path + "/" + ROIs_a_names[i] + ".npy")
            )
        ROIs_b_masks.append(np.load(
            quality_path + "/" + ROIs_b_names[i] + ".npy")
            )
    return  ComputeImageMetrics(
                ROIs_a=ROIs_a_masks,
                ROIs_b=ROIs_b_masks,
                emissions_a=ROIs_a_emissions,
                emissions_b=ROIs_b_emissions,
                names_a=ROIs_a_names,   
                names_b=ROIs_b_names)   