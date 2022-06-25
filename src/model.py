import sirf.STIR as pet
import os
import tensorboardX
import datetime
import socket
import torch

from .deep_image_prior import   normalize, \
                                ObjectiveFunctionModule, \
                                DeepImagePriorReconstructor

from .deep_image_prior.network import *



class ModelClass(object):
    def __init__(
            self, 
            cfg, 
            dataset):

        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        logdir = os.path.join('./', current_time + '_' + socket.gethostname())
        writer = tensorboardX.SummaryWriter(logdir=logdir)
        
        if cfg.model.name == 'baseline':
            baseline(
                cfg.model.num_subsets,
                cfg.model.num_epochs,
                dataset.objective_function,
                dataset.initial,
                dataset.quality_metrics,
                writer)

        elif cfg.model.name == 'unet':
            unetprior(
                cfg,
                dataset,
                writer)

        elif cfg.model.name == 'deepdecoder':
            deepdecoderprior(
                cfg,
                dataset,
                writer)

        else:
            raise NotImplementedError


def baseline(
        num_subsets, 
        num_epochs, 
        objective_function, 
        initial, 
        quality_metrics,
        writer):

    image = initial.get_uniform_copy(1)

    # CREATE RECONSTRUCTION OBJECT
    sirf_reconstruction = pet.OSMAPOSLReconstructor()
    sirf_reconstruction.set_objective_function(objective_function)
    num_subiterations = num_subsets*num_epochs
    sirf_reconstruction.set_num_subsets(num_subsets)                        
    sirf_reconstruction.set_num_subiterations(num_subiterations)

    # INITIALISE THE RECONSTRUCTION OBJECT
    sirf_reconstruction.set_up(image)
    sirf_reconstruction.set_current_estimate(initial)

    current_image = initial
    for i in range(0, num_subsets*num_epochs + 1):

        writer.add_image('recon', \
            normalize(current_image.as_array()), i)

        sirf_reconstruction.update(current_image)

        (crc, std) = \
            quality_metrics.get_all_metrics(current_image.as_array())

        writer.add_scalar('CRC', crc, i)
        writer.add_scalar('STDEV', std, i)
    writer.close()



def unetprior(
        cfg, 
        dataset, 
        writer):

    if cfg.model.torch_manual_seed:
            torch.random.manual_seed(cfg.model.torch_manual_seed)

    # Model
    model = UNet(
                1,
                1,
                channels=[128]*cfg.model.arch.scales,
                skip_channels=[0]*cfg.model.arch.scales,
                use_norm= cfg.model.arch.use_norm
                )
    
    # Input
    if cfg.model.random_input:
        input = 0.1 * torch.randn(1, * dataset.initial.shape)
    else:
        NotImplemented

    # Pre-trained
    if cfg.model.load_pretrain_model:
        path = cfg.model.learned_params_path
        model.load_state_dict(torch.load(path))

    obj_fun_module =    ObjectiveFunctionModule(
                            image_template=dataset.initial.get_uniform_copy(1), 
                            obj_fun = dataset.objective_function
                            )

    iterations = cfg.model.optim.iterations
    lr = cfg.model.optim.lr
    reconstructor = DeepImagePriorReconstructor(
                        model,
                        input,
                        obj_fun_module,
                        iterations,
                        lr,
                        writer)

    reconstructor.reconstruct(dataset.quality_metrics)


def deepdecoderprior(
        cfg, 
        dataset, 
        writer):

    if cfg.model.torch_manual_seed:
            torch.random.manual_seed(cfg.model.torch_manual_seed)

    # Model
    model = DeepDecoder(num_channels_up = [cfg.model.arch.channels]*5)
    
    # Input
    if cfg.model.random_input:
        input_shape = [1, cfg.model.arch.channels, 4, 4]

        input = torch.rand(input_shape, requires_grad=True)
    else:
        NotImplemented

    # Pre-trained
    if cfg.model.load_pretrain_model:
        path = cfg.model.learned_params_path
        model.load_state_dict(torch.load(path))

    obj_fun_module =    ObjectiveFunctionModule(
                            image_template=dataset.initial.get_uniform_copy(1), 
                            obj_fun = dataset.objective_function)

    iterations = cfg.model.optim.iterations

    lr = cfg.model.optim.lr

    reconstructor = DeepImagePriorReconstructor(
                        model,
                        input,
                        obj_fun_module,
                        iterations,
                        lr,
                        writer)

    reconstructor.reconstruct(
        dataset.quality_metrics,
        cfg.model.use_scheduler)