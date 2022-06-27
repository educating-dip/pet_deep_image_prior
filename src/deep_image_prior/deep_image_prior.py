import torch
import numpy as np
from tqdm import tqdm
from .utils import normalize

class DeepImagePriorReconstructor:
    def __init__(self, 
                    model,
                    input,
                    obj_fun_module,
                    iterations,
                    lr,
                    writer):

        self.device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.model = model.to(self.device)
        self.input = input.to(self.device)
        self.obj_fun_module = obj_fun_module.to(self.device)
        self.iterations = iterations
        self.lr = lr
        self.writer = writer


    def reconstruct(self, image_metrics, use_scheduler = False):

        self.init_optimizer()

        if use_scheduler:
            self.init_scheduler()

        best_loss = np.inf
        best_output = self.model(
            self.input
            ).detach()

        with tqdm(range(self.iterations), desc='PET-DIP') as pbar:
            for i in pbar:

                self.optimizer.zero_grad()
                output = self.model(self.input)

                loss = - torch.log(self.obj_fun_module(
                    output
                    )
                )
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

                self.writer.add_scalar('loss', loss.item(),  i)
                if i % 100 == 0:
                    self.writer.add_image('reco', normalize(
                        output[0, ...].detach().cpu().numpy() 
                        ), i)
                    if i  > 2500:
                        crc, stdev = image_metrics.get_all_metrics(
                            output[0, ...].detach().cpu().numpy()
                            )
                        for j in range(len(crc)):
                            self.writer.add_scalar(str(j) + '_CRC_' + image_metrics.names_a[j], crc[j], i)
                            self.writer.add_scalar(str(j) + '_STDEV_' + image_metrics.names_b[j], stdev[j], i)

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
