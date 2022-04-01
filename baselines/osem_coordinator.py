from builtins import breakpoint
import hydra
import torch
from omegaconf import DictConfig
from algos import osem
from dataset import (
        get_standard_acquisition_model,
        simulate, 
        get_data_sirf_standard_object
        )
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
            cfg, 
            return_sirf_obj=True
            )
        
        initial_image = image.get_uniform_copy(
                (example_image / cfg.data.gt_scl_fct * .6 ).max() / 4 
                )
        recon = osem(acquired_data, acquisition_model, initial_image, cfg.baseline, example_image)
        
    
        print('DIP reconstruction of sample {:d}'.format(i))
        print('PSNR:', PSNR(recon.as_array()[0], example_image.as_array()[0]))
        print('SSIM:', SSIM(recon.as_array()[0], example_image.as_array()[0]))

        import matplotlib.pyplot as plt 
        plt.imshow(recon.as_array()[0])
        plt.savefig('test.png')
        breakpoint()

if __name__ == '__main__':
    coordinator()