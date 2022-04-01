import os
import hydra
import torch
from omegaconf import DictConfig
from dataset import (
        get_standard_acquisition_model,
        simulate, 
        AcquisitionModelModule, 
        get_data_sirf_standard_object
        )
from deep_image_prior import DeepImagePriorReconstructor
from deep_image_prior.utils import PSNR, SSIM


@hydra.main(config_path='../cfgs', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    image, attn_image, data_template = get_data_sirf_standard_object(cfg)
    acquisition_model = get_standard_acquisition_model(
        image,
        data_template,
        cfg,
        attn_image = attn_image
        )

    for i in range(cfg.num_images):

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed + i)  # for reproducible noise in simulate
        
        acquired_data, example_image = simulate(
            image,
            data_template, 
            acquisition_model,
            cfg
            )

        acq_model_module = AcquisitionModelModule(
            image.clone(),
            data_template.clone(),
            acquisition_model
            )

        reconstructor = DeepImagePriorReconstructor(acq_model_module, image, cfg=cfg)
        recon, _ = reconstructor.reconstruct(            
                    acquired_data.to(reconstructor.device),
                    ground_truth = example_image.to(reconstructor.device)
                )

        torch.save(reconstructor.model.state_dict(),
                './dip_model_{}.pt'.format(i))

        print('DIP reconstruction of sample {:d}'.format(i))
        print('PSNR:', PSNR(recon, example_image[0, 0].cpu().numpy()))
        print('SSIM:', SSIM(recon, example_image[0, 0].cpu().numpy()))

if __name__ == '__main__':
    coordinator()