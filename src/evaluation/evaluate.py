'''
Developed from scrtach by Mert Sengul.
Please cite the repo if you readapt.
'''

import torch
from tqdm import tqdm
import numpy as np
import os

class Evaluater:
    '''
    Evaluater object.
    '''
    def __init__(
        self,
        model,
        criterion,
        checkpoint,
        experiment,
        device,
    ):

        super().__init__()

        self.model = model
        self.criterion = criterion
        self.checkpoint = checkpoint
        self.experiment = experiment
        self.device = device
    
    
    def evaluate(self, holdout_loader):
        # evaluate
        eval_loss, eval_rc_loss, eval_kl_loss = self._evaluate(holdout_loader)
        print('Loss: {} - Reconst Loss: {} - KL Loss: {}'.format(eval_loss, eval_rc_loss, eval_kl_loss))

    def evaluate_data(self, data):
        self.model.load_state_dict(torch.load(self.checkpoint, map_location=self.device))
        
        # put model in evaluation mode
        self.model.eval()

        with torch.no_grad():
            images, y = data
            images = images.to(self.device)
            y = y.to(self.device)            
            reconstructed_image = self.model(images)
            loss = self.criterion(reconstructed_image, y)
            

        print('Loss: {}'.format(loss.detach().cpu().item()))
        return images, reconstructed_image

    def _evaluate(self, loader):
        _loss = []
       
        self.model.eval()

        with torch.no_grad():
            for images, y in tqdm(loader):   
                #images, _ = data # No need to return
                images = images.to(self.device)
                y = y.to(self.device)
                
                reconstructed_image = self.model(images)
                
                loss = self.criterion(reconstructed_image, y)
                _loss.append(loss.detach().cpu().item())


        epoch_loss = np.mean(_loss)

        return epoch_loss