import sirf.STIR as pet
import os
import tensorboardX
import datetime
import socket
import torch
import numpy as np
import math

from .deep_image_prior import   normalize, \
                                ObjectiveFunctionModule, \
                                DeepImagePriorReconstructor, \
                                DPDeepImagePriorReconstructor, \
                                PETAcquisitionModelModule

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
                cfg.model.eta,
                dataset.objective_function,
                dataset.initial,
                dataset.quality_metrics,
                dataset.sensitivity_image,
                writer)

        elif cfg.model.name == 'unet':
            unetprior(
                cfg,
                dataset,
                writer)
        
        elif cfg.model.name == 'dp_unet':
            dpunetprior(
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

def herman_meyer_order(n):
    # Assuming that the subsets are in geometrical order
    n_variable = n
    i = 2
    factors = []
    while i * i <= n_variable:
        if n_variable % i:
            i += 1
        else:
            n_variable //= i
            factors.append(i)
    if n_variable > 1:
        factors.append(n_variable)
    n_factors = len(factors)
    order =  [0 for _ in range(n)]
    value = 0
    for factor_n in range(n_factors):
        n_rep_value = 0
        if factor_n == 0:
            n_change_value = 1
        else:
            n_change_value = math.prod(factors[:factor_n])
        for element in range(n):
            mapping = value
            n_rep_value += 1
            if n_rep_value >= n_change_value:
                value = value + 1
                n_rep_value = 0
            if value == factors[factor_n]:
                value = 0
            order[element] = order[element] + math.prod(factors[factor_n+1:]) * mapping
    return order

def baseline(
        num_subsets, 
        num_epochs,
        gamma,
        objective_function, 
        initial, 
        quality_metrics,
        sensitivity_image,
        writer):

    image = initial.get_uniform_copy(1)

    # CREATE RECONSTRUCTION OBJECT
    objective_function.set_num_subsets(num_subsets)
    # INITIALISE THE RECONSTRUCTION OBJECT
    objective_function.set_up(image)

    x_k = initial
    delta = initial.clone().fill(1e-9)
    alpha_k = 1
    outside_fov = sensitivity_image.as_array() == 0
    ordered_subsets = herman_meyer_order(num_subsets)
    print(f"Subset order: {ordered_subsets}")
    for i in range(num_epochs):
        preconditioner = ((x_k+delta)/(sensitivity_image))
        tmp = preconditioner.as_array()
        tmp[outside_fov] = 0
        preconditioner.fill(tmp)
        alpha_k = 1/(gamma*i+1)
        print(f"Epoch {i+1}")
        print(f"Precond norm {preconditioner.norm()}")
        print(f"Alpha k is {alpha_k}")
        if np.isnan(x_k.as_array()).any():
            print(f"Eta value is {gamma}")
            break
        for j in range(num_subsets):
            ssg = objective_function.get_subset_gradient(x_k, ordered_subsets[j])
            precond_ssg = preconditioner * num_subsets * ssg
            if gamma == 0:
                precond_ssg = preconditioner * ssg
            x_k = x_k + alpha_k * precond_ssg
            print(f"SSGradient norm {ssg.norm()}")
            tmp = x_k.as_array()
            tmp[tmp<0] = 0
            x_k.fill(tmp)
        obj_val = objective_function.value(x_k)
        writer.add_scalar('OBJ_FUNC', obj_val, i + 1)
        (crc, stdev) = \
            quality_metrics.get_all_metrics(x_k.as_array())
        writer.add_scalar(str(0) + '_STDEV_' + quality_metrics.names_b[0], stdev[0], i + 1)
        for k in range(len(crc)):
            writer.add_scalar(str(k) + '_CRC_' + quality_metrics.names_a[k], crc[k], i + 1)
        """ if x_k_1.as_array().shape[0] == 1:
            writer.add_image('recon', \
                normalize(x_k_1.as_array()), i)
        elif x_k_1.as_array().shape[0] > 1:
            writer.add_image('recon', \
                normalize(x_k_1.as_array()[[5],...]), i) """
        x_k.write(f'Volume_epoch_{i+1}.hv')
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

def dpunetprior(
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

    device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))
    data = torch.tensor(dataset.prompts.as_array()).to(device)
    acq_model_module =   PETAcquisitionModelModule(
                                image_template = dataset.initial, 
                                data_template = dataset.prompts, 
                                acq_model = dataset.acquisition_model)

    iterations = cfg.model.optim.iterations
    lr = cfg.model.optim.lr

    const = data*torch.log(data+1e-8)
    const = const.sum().to(device)
    loss = torch.nn.PoissonNLLLoss(log_input=False,reduction='sum')
    func = lambda x: loss(x, data) - x.detach().sum() + const
    dp_approx = data.shape[-1]*data.shape[-2]/2
    dp_nearly_exact_fnc = lambda x: dp_approx + torch.sum((x**2+2.5702*x-1.5205)/ \
        (12 * x**3 - 5.6244*x**2 + 17.9347*x + 3.0410))
    dp_nearly_exact = dp_nearly_exact_fnc(data)
    reconstructor = DPDeepImagePriorReconstructor(
        model,
        input,
        acq_model_module,
        func,
        iterations,
        lr,
        writer,
        dp_approx,
        dp_nearly_exact)

    reconstructor.reconstruct(dataset.quality_metrics)