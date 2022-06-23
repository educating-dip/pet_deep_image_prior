import sirf.STIR as pet
import os
import tensorboardX
import datetime
import socket

from .deep_image_prior import normalize, ObjectiveFunctionModule, DeepImagePriorReconstructor, DeepDecoderPriorReconstructor

class ModelClass(object):
    def __init__(self, cfg, dataset):
        
        if cfg.model.name == 'baseline':
            model = baseline(
                    cfg.model.num_subsets,
                    cfg.model.num_epochs,
                    dataset.objective_function,
                    dataset.initial,
                    dataset.quality_metrics
                    )
        elif cfg.model.name == 'unet':
            reconstructor = DeepImagePriorReconstructor(
            obj_fun_module = ObjectiveFunctionModule(
                image_template=dataset.initial.get_uniform_copy(1), 
                obj_fun = dataset.objective_function
                ), 
                image_template=dataset.initial,
                cfgs=cfg
                )
            reconstructor.reconstruct(
                dataset.quality_metrics
            )
        elif cfg.model.name == 'deepdecoder':
            reconstructor = DeepDecoderPriorReconstructor(
            obj_fun_module = ObjectiveFunctionModule(
                image_template=dataset.initial.get_uniform_copy(1), 
                obj_fun = dataset.objective_function
                ), 
                image_template=dataset.initial,
                cfgs=cfg
                )
            reconstructor.reconstruct(
                dataset.quality_metrics
            )
            
            raise NotImplementedError
        elif cfg.model.name == 'deepdecoder':
            raise NotImplementedError
        else:
            raise NotImplementedError






def baseline(num_subsets, num_epochs, objective_function, initial, quality_metrics):
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

    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    logdir = os.path.join('./', current_time + '_' + socket.gethostname())
    writer = tensorboardX.SummaryWriter(logdir=logdir)


    # TO DO GET THE QUALITY METRICS
    current_image = initial
    for i in range(0, num_subsets*num_epochs + 1):

        writer.add_image('recon', 
            normalize(
                current_image.as_array()
            ), i)

        sirf_reconstruction.update(
            current_image
            )

        (crc, std) = quality_metrics.get_all_metrics(
            current_image.as_array()
            )

        writer.add_scalar('CRC', crc, i)
        writer.add_scalar('STDEV', std, i)

    """ if cfg.dataset.name == '2D': 
        row_lesion = 139
        np.save('profile', current_image.as_array()[0, row_lesion, :])

    np.save('recon', current_image.as_array()) """
    writer.close()

