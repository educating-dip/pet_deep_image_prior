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
                    writer,
                    every_iter_save = False):

        self.device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.model = model.to(self.device)
        self.input = input.to(self.device)
        self.obj_fun_module = obj_fun_module.to(self.device)
        self.iterations = iterations
        self.lr = lr
        self.writer = writer
        self.every_iter_save = every_iter_save


    def reconstruct(self, image_metrics, use_scheduler = False):

        self.init_optimizer()

        if use_scheduler:
            self.init_scheduler()

        best_loss = np.inf
        with torch.no_grad(): 
            best_output = self.model(self.input)

        if self.every_iter_save:
            counter = 0
            n_dump = 1000
            save_tensor = torch.zeros(n_dump,1,47,128,128)
            epoch_vector = torch.zeros(n_dump)
        with tqdm(range(self.iterations)) as pbar:
            for i in pbar:

                self.optimizer.zero_grad()
                output = self.model(self.input)

                loss = - self.obj_fun_module(output)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                self.optimizer.step()

                if self.every_iter_save:
                    counter = counter + 1
                    save_tensor[counter-1,...] = output.detach().cpu()
                    epoch_vector[counter-1] = i
                    if counter == n_dump:
                        torch.save({"epochs": epoch_vector, "tensor":save_tensor}, f'Save_all_iter_{i}.pt')
                        save_tensor = torch.zeros(n_dump,1,47,128,128)
                        epoch_vector = torch.zeros(n_dump)
                        counter = 0

                if use_scheduler:
                    self.writer.add_scalar("LR", self.scheduler.get_last_lr(), i)
                    self.scheduler.step()

                for p in self.model.parameters():
                    p.data.clamp_(-1000, 1000) # MIN,MAX

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_output = output.detach()

                            
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_output = output.detach()
                    

                self.writer.add_scalar('loss', loss.item(),  i)
                pbar.set_description("PET-DIP loss {:.3E}".format(loss.item()))
                if i % 1000 == 0:
                    if i  > 1000:
                        crc, stdev = image_metrics.get_all_metrics(
                            output[0, 0,...].detach().cpu().numpy()
                            )
                        for j in range(len(crc)):
                            self.writer.add_scalar(str(j) + '_CRC_' + image_metrics.names_a[j], crc[j], i)
                            self.writer.add_scalar(str(j) + '_STDEV_' + image_metrics.names_b[j], stdev[j], i)
                if i % 1000 == 0:
                    if i  > 1000:
                        torch.save(best_output[0, 0,...],f"Best_volume_epoch_{i}.torch")
                        torch.save({
                            'epoch': i,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': loss,
                            }, f"checkpoint_epoch_{i}.torch")
            

        torch.save(best_output[0, 0,...],f"Final_best_volume.torch")
        torch.save({
            'epoch': i,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            }, f"Final_checkpoint.torch")
        self.writer.close()
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
