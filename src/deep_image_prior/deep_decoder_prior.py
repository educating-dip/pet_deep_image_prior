import os
import socket
import datetime
import torch
import numpy as np
import tensorboardX
from hydra.utils import get_original_cwd
from tqdm import tqdm
from .network import UNet
from .utils import normalize
from copy import deepcopy

class DeepImagePriorReconstructor():

    def __init__(self, obj_fun_module, image_template, cfgs):

        self.cfgs = cfgs
        self.image_template = image_template
        self.device = torch.device(
            ('cuda:0' if torch.cuda.is_available() else 'cpu')
            )
        self.obj_fun_module = obj_fun_module.to(self.device)
        self.init_model()

    def init_model(self):

        self.model = UNet(
            1,
            1,
            channels=[128]*self.cfgs.net.arch.scales,
            skip_channels=[0]*self.cfgs.net.arch.scales,
            use_norm=self.cfgs.net.arch.use_norm
            ).to(self.device)

        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        logdir = os.path.join(
            self.cfgs.net.log_path,
            current_time + '_' + socket.gethostname()
            )
        self.writer = tensorboardX.SummaryWriter(logdir=logdir)

    def reconstruct(self, image_metrics, init_model=True):

        if self.cfgs.net.torch_manual_seed:
            torch.random.manual_seed(self.cfgs.net.torch_manual_seed)

        if init_model: 
            self.init_model()
        if self.cfgs.net.load_pretrain_model:
            path = os.path.join(
                get_original_cwd(),
                self.cfgs.net.learned_params_path if self.cfgs.net.learned_params_path.endswith('.pt') \
                    else self.cfgs.net.learned_params_path + '.pt')
            self.model.load_state_dict(
                torch.load(
                    path, map_location=self.device
                    )
                )
        else:
            self.model.to(self.device)

        self.model.train()
        self.net_input = 0.1 * \
            torch.randn(
                1, *self.image_template.shape
            ).to(self.device)

        self.init_optimizer()

        best_loss = np.inf
        best_output = self.model(
            self.net_input
            ).detach()

        with tqdm(range(self.cfgs.net.optim.iterations), desc='PET-DIP', disable=not self.cfgs.net.show_pbar) as pbar:
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

                for p in self.model.parameters():
                    p.data.clamp_(-1000, 1000) # MIN,MAX
            
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_output = output.detach()

                self.writer.add_scalar('loss', loss.item(),  i)
                if i % 100 == 0:
                    self.writer.add_image('reco', normalize(
                        output[0, ...].detach().cpu().numpy() 
                        ), i)
                    if i  > 2500:
                        crc, stdev = image_metrics.get_all_metrics(
                            output[0, ...].detach().cpu().numpy()
                            )
                        self.writer.add_scalar('crc', crc, i)
                        self.writer.add_scalar('stdev', stdev, i)

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

        self._optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfgs.net.optim.lr)

    @property
    def optimizer(self):
        """
        :class:`torch.optim.Optimizer` :
        The optimizer, usually set by :meth:`init_optimizer`, which gets called
        in :meth:`train`.
        """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value