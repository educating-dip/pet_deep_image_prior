import os
import sirf.STIR as pet
import sirf.STIR
import datetime 
import socket
import numpy
import tensorboardX
from deep_image_prior import normalize, PSNR

sirf.STIR.set_verbosity(False)

def osem(acquired_data, acquisition_model, initial_image, cfg, ground_truth=None):

    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    comment = 'OSEM'
    logdir = os.path.join(
        './',
        current_time + '_' + socket.gethostname() + comment)
    writer = tensorboardX.SummaryWriter(logdir=logdir)
    
    if cfg.obj_fun == 'poisson': 
        obj_fun = pet.make_Poisson_loglikelihood(acquired_data)
        obj_fun.set_acquisition_model(acquisition_model)
    else: 
        raise NotImplementedError
    
    OSMAPOSL = pet.OSMAPOSLReconstructor()
    OSMAPOSL.set_num_subsets(cfg.impl.num_subsets)
    OSMAPOSL.set_num_subiterations(1)
    OSMAPOSL.set_objective_function(obj_fun)
    if cfg.quadratic_prior.active:  
        prior = pet.QuadraticPrior()
        print('using Quadratic prior...')
        prior.set_up(
            acquired_data.create_uniform_image(1.0)
            )
        prior.set_penalisation_factor(
            float(cfg.quadratic_prior.gamma)
            )
        obj_fun.set_prior(prior)

    curr_image = initial_image.clone()
    OSMAPOSL.set_current_estimate(curr_image)
    OSMAPOSL.set_up(curr_image)
    for i in range(1, cfg.impl.num_subiters + 1):
        OSMAPOSL.update(curr_image)
        writer.add_image('recon', normalize(curr_image.as_array()), i)
    
        if ground_truth is not None:
            output_psnr = PSNR(curr_image.as_array()[0], ground_truth.as_array()[0])
            writer.add_scalar('output_psnr', output_psnr, i)

    writer.close()

    return curr_image