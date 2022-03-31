import os
import sirf.STIR as pet
import sirf.STIR
sirf.STIR.set_verbosity(False)
import numpy as np
import torch

def get_standard_acquisition_model(image, data_template, cfg, attn_image=None):

    acquisition_model = pet.AcquisitionModelUsingRayTracingMatrix()
    acquisition_model.set_num_tangential_LORs(cfg.acquisition_model.set_num_tangential_LORs)

    if attn_image is not None:
        acquisition_model_for_attn = pet.AcquisitionModelUsingRayTracingMatrix()
        asm_attn = pet.AcquisitionSensitivityModel(attn_image, acquisition_model_for_attn)
        asm_attn.set_up(data_template)
        attn_factors = asm_attn.forward(data_template.get_uniform_copy(1.0))
        asm_attn = pet.AcquisitionSensitivityModel(
            attn_factors * cfg.acquisition_model.asm_attn.scl_fct)
        acquisition_model.set_acquisition_sensitivity(asm_attn)
    
    if cfg.acquisition_model.add_background: 
        acquisition_model.set_background_term(data_template.get_uniform_copy())

    return acquisition_model

def simulate(image, data_template, acquisition_model, cfg):

    acquisition_model.set_up(data_template, image)
    acquired_data = acquisition_model.forward(image)
    noisy_acquired_data_array = np.random.poisson(
        cfg.data.poisson_noise.scl_fct * acquired_data.clone().as_array()
        ).astype('float64')
    acquired_data = acquired_data.clone()
    acquired_data.fill(noisy_acquired_data_array)
    return torch.from_numpy(acquired_data.as_array()), torch.from_numpy(image.as_array())


def get_2D_data_sirf_standard_object(cfg):

    image = pet.ImageData(
        os.path.join(cfg.data.path, 'emission.hv')
        )
    image_array = image.as_array()
    image_array *= cfg.data.gt_scl_fct
    image.fill(image_array)
    attn_image = pet.ImageData(
        os.path.join(cfg.data.path, 'attenuation.hv')
        )
    data_template = pet.AcquisitionData(
        os.path.join(cfg.data.path, 'template_sinogram.hs')
        )
    return image, attn_image, data_template
