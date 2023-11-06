'''
Developed from scratch by Mert Sengul.
Please cite the repo if you readapt.
'''

import torch
from tqdm import tqdm
import numpy as np
import os
import wandb

class Trainer:
    '''
    Trainer object.
    '''
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        checkpoint,
        experiment,
        device,
        latent_dim,
        
    ):

        super().__init__()

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.checkpoint = checkpoint
        self.experiment = experiment
        self.device = device
        self.latent_dim = latent_dim
    
    def fit(self, train_loader, val_loader, epochs):
        self._wandb_initiate()
        for epoch in tqdm(range(epochs)):
            # train
            train_loss, train_rc_loss, train_kl_loss = self._train(train_loader)
            #print('Epoch: {} - Loss: {} - Reconst Loss: {} - KL Loss: {}'.format(epoch, train_loss, train_rc_loss, train_kl_loss))
            wandb.log({"Training Loss": train_loss, "Training Reconst Loss": train_rc_loss, "Training KL Loss": train_kl_loss})
            # validate
            val_loss, val_rc_loss, val_kl_loss = self._validate(val_loader)
            #print('Epoch: {} - Loss: {} - Reconst Loss: {} - KL Loss: {}'.format(epoch, val_loss, val_rc_loss, val_kl_loss))
            wandb.log({"Validation Loss": val_loss, "Validation Reconst Loss": val_rc_loss, "Validation KL Loss": val_kl_loss})
            #save model state
            self._save_checkpoint(train_loss, val_loss, epoch)

    def _wandb_initiate(self):
        #### W&B INIT ###
        wandb.init(project=self.experiment)
        wandb.watch(
            self.model,
            criterion=self.criterion,
            log= 'parameters',#Optional[Literal['gradients', 'parameters', 'all']] = "gradients",
            log_freq=100,
            log_graph=True
            )
        ###############

    def _save_checkpoint(self, train_loss, val_loss, epoch):
        path = '{}/{}'.format(self.checkpoint, self.experiment)
        if not os.path.isdir(path):
            os.mkdir(path)
        
        torch.save(self.model.state_dict(), '{}/checkpoint_{}.pth'.format(path, epoch))

    def _train(self, loader):
        _loss = []
        self.model.to(self.device)
        self.model.train()

        for i, data in tqdm(enumerate(loader)):            
            self.optimizer.zero_grad()

            images, _ = data # No need to return
            images = images.to(self.device)
            
            reconstructed_image = self.model(points)
            
            loss = self.criterion(reconstructed_image, images)
            loss.backward()
            self.optimizer.step()

            _loss.append(loss.item())

        epoch_loss = np.mean(_loss)
        
        return epoch_loss

    def _validate(self, loader):
        _loss = []
        self.model.eval()

        with torch.no_grad():
            for i, data in tqdm(enumerate(loader)):   
                images, _ = data # No need to return
                images = images.to(self.device)
                reconstructed_image = self.model(points)
                
                loss = self.criterion(reconstructed_image, images)
                _loss.append(loss.item())


        epoch_loss = np.mean(_loss)

        return epoch_loss
        

    