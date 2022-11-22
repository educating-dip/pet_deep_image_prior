import torch
import numpy as np
from tqdm import tqdm
from .utils import normalize

class DPDeepImagePriorReconstructor:
    def __init__(
            self,
            model,
            input,
            acq_model,
            loss,
            iterations,
            lr,
            writer,
            dp_approx,
            dp_nearly_exact):

        self.device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.model = model.to(self.device)
        self.input = input.to(self.device)
        self.acq_model = acq_model.to(self.device)
        self.loss = loss
        self.iterations = iterations
        self.lr = lr
        self.writer = writer
        self.dp_approx = dp_approx
        self.dp_nearly_exact = dp_nearly_exact


    def reconstruct(self, image_metrics, use_scheduler = False):

        self.init_optimizer()

        if use_scheduler:
            self.init_scheduler()

        best_loss = np.inf
        best_output = self.model(
            self.input
            )[0].detach()

        dp_solns = 10
        dp_approx_i = 1
        dp_nearly_exact_i = 1

        with tqdm(range(self.iterations)) as pbar:
            for i in pbar:
                self.optimizer.zero_grad()
                output = self.model(self.input)[0]
                proj = self.acq_model(output)
                loss = self.loss(proj)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                self.optimizer.step()

                if use_scheduler:
                    self.scheduler.step()
                for p in self.model.parameters():
                    p.data.clamp_(-1000, 1000) # MIN,MAX
            
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_output = output.detach()
                self.writer.add_scalar('loss',loss.item(),i)

                if dp_approx_i < dp_solns and loss.item() < self.dp_approx:
                    self.writer.add_scalar('dp_approx_loss', loss.item(), i)
                    self.writer.add_image('dp_approx_reco', normalize(
                        output.detach().cpu().numpy() 
                        ), i)
                    dp_approx_i += 1

                
                if dp_nearly_exact_i < dp_solns and loss.item() < self.dp_nearly_exact:
                    self.writer.add_scalar('dp_nearly_exactx_loss', loss.item(), i)
                    self.writer.add_image('dp_nearly_exact_reco', normalize(
                        output.detach().cpu().numpy() 
                        ), i)
                    dp_nearly_exact_i += 1

                if i > 10:
                    if i % 10 == 0:
                        self.writer.add_image('reco', normalize(
                            output.detach().cpu().numpy() 
                            ), i)
                    crc, stdev = image_metrics.get_all_metrics(
                                output.detach().cpu().numpy()
                                )
                    for j in range(len(crc)):
                        self.writer.add_scalar(str(j) + '_CRC_' + image_metrics.names_a[j], crc[j], i)
                        self.writer.add_scalar(str(j) + '_STDEV_' + image_metrics.names_b[j], stdev[j], i)
                
        np.save('recon', best_output[0,0,...].detach().cpu().numpy())
        return best_output[0, 0, ...].cpu().numpy()

    def init_optimizer(self):
        """
        Initialize the optimizer.
        """

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def init_scheduler(self):
        """
        Initialize the scheduler.
        """
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 
                                                                    self.iterations, 
                                                                    eta_min=0, 
                                                                    last_epoch=-1, 
                                                                    verbose=False)
