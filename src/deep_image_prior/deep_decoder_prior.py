import os
import socket
import datetime
import torch
import numpy as np
import tensorboardX
from hydra.utils import get_original_cwd
from tqdm import tqdm
from .network import DeepDecoder
from .utils import normalize
from copy import deepcopy

class DeepDecoderPriorReconstructor():

    def __init__(self, obj_fun_module, image_template, cfgs):

        self.cfgs = cfgs
        self.image_template = image_template
        self.device = torch.device(
            ('cuda:0' if torch.cuda.is_available() else 'cpu')
            )
        self.obj_fun_module = obj_fun_module.to(self.device)
        self.init_model()

    def init_model(self):

        self.model = DeepDecoder(num_channels_up = [self.cfgs.model.arch.channels]*5).to(self.device)
        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        logdir = os.path.join(
            self.cfgs.model.log_path,
            current_time + '_' + socket.gethostname()
            )
        self.writer = tensorboardX.SummaryWriter(logdir=logdir)

    def reconstruct(self, image_metrics, init_model=True):

        if self.cfgs.model.torch_manual_seed:
            torch.random.manual_seed(self.cfgs.model.torch_manual_seed)

        if init_model: 
            self.init_model()
            
        if self.cfgs.model.load_pretrain_model:
            path = os.path.join(
                get_original_cwd(),
                self.cfgs.model.learned_params_path if self.cfgs.model.learned_params_path.endswith('.pt') \
                    else self.cfgs.model.learned_params_path + '.pt')
            self.model.load_state_dict(
                torch.load(
                    path, map_location=self.device
                    )
                )
        else:
            self.model.to(self.device)

        self.model.train()
        input_shape = [1, self.cfgs.model.arch.channels, 4, 4]

        self.net_input = torch.rand(input_shape, 
            generator = torch.Generator().manual_seed(0),
            requires_grad=True).to(self.device)

        self.init_optimizer()
        self.init_scheduler()

        best_loss = np.inf

        best_output = self.model(
            self.net_input
            ).detach()

        with tqdm(range(self.cfgs.model.optim.iterations), desc='PET-DIP', disable=not self.cfgs.model.show_pbar) as pbar:
            for i in pbar:

                self.optimizer.zero_grad()
                output = self.model(self.net_input)

                loss = - torch.log(self.obj_fun_module(
                    output
                    )
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                self.optimizer.step()
                self.scheduler.step()
            
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_output = output.detach()

                self.writer.add_scalar('loss', loss.item(),  i)
                if i % 25 == 0:
                    self.writer.add_image('reco', normalize(
                        output[0, ...].detach().cpu().numpy() 
                        ), i)
                    if i  > 500:
                        crc, stdev = image_metrics.get_all_metrics(
                            output[0, ...].detach().cpu().numpy()
                            )
                        self.writer.add_scalar('crc', crc, i)
                        self.writer.add_scalar('stdev', stdev, i)
                        self.writer.add_scalar('lr', self.scheduler.get_lr(), i)

        self.writer.close()
        
        crc, stdev = image_metrics.get_all_metrics(
            best_output[0, ...].detach().cpu().numpy()
            )
        row_lesion = 139
        np.save('recon', best_output[0,0,...].detach().cpu().numpy())
        np.save('profile', best_output[0,0, row_lesion, :].detach().cpu().numpy())
        np.save('crc',crc)
        np.save('std_dev',stdev)

        return best_output[0, 0, ...].cpu().numpy()

    def init_optimizer(self):
        """
        Initialize the optimizer.
        """

        self._optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfgs.model.optim.lr)

    def init_scheduler(self):
        """
        Initialize the scheduler.
        """
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer, self.cfgs.model.optim.iterations, eta_min=0, last_epoch=- 1, verbose=False)
        

    @property
    def optimizer(self):
        """
        :class:`torch.optim.Optimizer` :
        The optimizer, usually set by :meth:`init_optimizer`, which gets called
        in :meth:`train`.
        """
        return self._optimizer
    @property
    def scheduler(self):
        """
        :class:`torch.optim.Scheduler` :
        The scheduler, usually set by :meth:`init_scheduler`, which gets called
        in :meth:`train`.
        """
        return self._scheduler

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value