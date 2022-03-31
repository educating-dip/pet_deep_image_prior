import os
import socket
import datetime
import torch
import numpy as np
import tensorboardX
from torch.optim import Adam
from torch.nn import MSELoss, PoissonNLLLoss
from hydra.utils import get_original_cwd
from tqdm import tqdm
from warnings import warn 
from .network import UNet
from .utils import tv_loss, PSNR, normalize
from copy import deepcopy

class DeepImagePriorReconstructor():
    """
    CT reconstructor applying DIP with TV regularization (see [2]_).
    The DIP was introduced in [1].
    .. [1] V. Lempitsky, A. Vedaldi, and D. Ulyanov, 2018, "Deep Image Prior".
           IEEE/CVF Conference on Computer Vision and Pattern Recognition.
           https://doi.org/10.1109/CVPR.2018.00984
    .. [2] D. Otero Baguer, J. Leuschner, M. Schmidt, 2020, "Computed
           Tomography Reconstruction Using Deep Image Prior and Learned
           Reconstruction Methods". Inverse Problems.
           https://doi.org/10.1088/1361-6420/aba415
    """

    def __init__(self, acquisition_model_module, image_template, cfg):

        self.cfg = cfg
        self.image_template = image_template
        self.device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.acquisition_model_module = acquisition_model_module.to(self.device)
        self.init_model()

    def init_model(self):

        input_depth = 1
        output_depth = 1
        self.model = UNet(
            input_depth,
            output_depth,
            channels=self.cfg.net.arch.channels[:self.cfg.net.arch.scales],
            skip_channels=self.cfg.net.arch.skip_channels[:self.cfg.net.arch.scales],
            use_relu=self.cfg.net.arch.use_relu,
            use_norm=self.cfg.net.arch.use_norm
            ).to(self.device)

        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        comment = 'DIP+TV'
        logdir = os.path.join(
            self.cfg.net.log_path,
            current_time + '_' + socket.gethostname() + comment)
        self.writer = tensorboardX.SummaryWriter(logdir=logdir)

    def reconstruct(self, noisy_observation, fbp=None, ground_truth=None, init_model=True, use_tv_loss=False):

        if self.cfg.net.torch_manual_seed:
            torch.random.manual_seed(self.cfg.net.torch_manual_seed)

        if init_model: 
            self.init_model()

        if self.cfg.net.load_pretrain_model:
            path = os.path.join(
                get_original_cwd(),
                self.cfg.net.learned_params_path if self.cfg.net.learned_params_path.endswith('.pt') \
                    else self.cfg.net.learned_params_path + '.pt')
            self.model.load_state_dict(torch.load(path, map_location=self.device))
        else:
            self.model.to(self.device)

        self.model.train()

        if self.cfg.net.recon_from_randn:
            self.net_input = 0.1 * \
                torch.randn(1, *self.image_template.shape).to(self.device)
        else:
            self.net_input = fbp.to(self.device)

        self.init_optimizer()
        y_delta = noisy_observation.to(self.device)

        criterion = PoissonNLLLoss(log_input=False)

        best_loss = np.inf
        model_out = self.model(self.net_input)
        best_output, pre_activation_best_output = model_out[0].detach(), model_out[1].detach()
        best_params_state_dict = deepcopy(self.model.state_dict())

        with tqdm(range(self.cfg.net.optim.iterations), desc='PET-DIP', disable= not self.cfg.net.show_pbar) as pbar:
            for i in pbar:
                self.optimizer.zero_grad()
                model_out = self.model(self.net_input)
                output, pre_activation_output = model_out[0], model_out[1].detach()
                proj_output = self.acquisition_model_module(output)
                loss = criterion(proj_output, y_delta) 
                if use_tv_loss: 
                    loss = loss + self.cfg.net.optim.gamma * tv_loss(output)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                if loss.item() < best_loss:
                    best_params_state_dict = deepcopy(self.model.state_dict())
                self.optimizer.step()

                for p in self.model.parameters():
                    p.data.clamp_(-1000, 1000) # MIN,MAX

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_output = output.detach()
                    pre_activation_best_output = pre_activation_output

                if ground_truth is not None:
                    best_output_psnr = PSNR(best_output.detach().cpu() / self.cfg.data.poisson_noise.scl_fct, ground_truth.cpu())
                    output_psnr = PSNR(output.detach().cpu() / self.cfg.data.poisson_noise.scl_fct, ground_truth.cpu())
                    pbar.set_description('DIP output_psnr={:.1f}'.format(output_psnr), refresh=False)
                    self.writer.add_scalar('best_output_psnr', best_output_psnr, i)
                    self.writer.add_scalar('output_psnr', output_psnr, i)

                self.writer.add_scalar('loss', loss.item(),  i)
                if i % 100 == 0:
                    self.writer.add_image('reco', normalize(best_output[0, ...]).cpu().numpy(), i)

        self.model.load_state_dict(best_params_state_dict)
        self.writer.close()

        return best_output[0, 0, ...].cpu().numpy() / self.cfg.data.poisson_noise.scl_fct, pre_activation_best_output[0, 0, ...].cpu().numpy()

    def init_optimizer(self):
        """
        Initialize the optimizer.
        """

        self._optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.net.optim.lr)

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