import os
import numpy as np
from torch import Tensor, nn, optim
import torch
from torch.utils.data import DataLoader
from utils.conf import Configuration
from utils.sample import getSamples, TSData
from model.seanet import SEAnet
from utils.init import LSUVinit
from model.loss import ScaledReconsLoss, ScaledTransLoss
from model.normalization import getSRIPTerm
import logging


class Experiment:
    
    def __init__(self, conf:Configuration):
        self.conf = conf
        self.epoch_max = conf.getEntry("epoch_max")
        self.device = conf.getEntry("device")
        self.log_path = conf.getEntry("log_path")
        self.model_path = self.conf.getEntry("model_path")
        
        logging.basicConfig(
            level = logging.INFO,
            format = '%(asctime)s - %(levelname)s - %(message)s',
            filename = self.log_path,
            filemode = "w"
        )
        
        logging.info(f"Experiment initialized with max epochs: {self.epoch_max} on device: {self.device}")
        
        
    def setup(self) -> None:
        batch_size = self.conf.getEntry("batch_size")
        dim_series = self.conf.getEntry("dim_series")
        dim_embedding = self.conf.getEntry("dim_embedding")
        
        train_samples, val_samples = getSamples(self.conf)
        
        self.train_loader = DataLoader(TSData(train_samples), batch_size=batch_size, shuffle=True)
        self.train_query_loader = DataLoader(TSData(train_samples), batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(TSData(val_samples), batch_size=batch_size, shuffle=True)
        self.val_query_loader = DataLoader(TSData(val_samples), batch_size=batch_size, shuffle=True)
        
        self.model = SEAnet(self.conf)
        
        if os.path.exists(self.conf.getEntry("model_path")):
            logging.info("Model loading...")
            self.model.load_state_dict(torch.load(self.model_path))
        else:
            logging.info("Model initializing...")
            self.model = self.initModel(self.model, val_samples)
        
        self.trans_loss_calculator = ScaledTransLoss(dim_series, dim_embedding, to_scale=True).to(self.device)
        self.recons_loss_calculator = ScaledReconsLoss(dim_series, to_scale=True).to(self.device)
        
        self.optimizer = self.getOptimizer()
        
        self.orth_regularizer = self.conf.getEntry('orth_regularizer')
        if self.orth_regularizer == 'srip':
            if self.conf.getEntry('srip_mode') == 'fix':
                self.srip_weight = self.conf.getEntry('srip_cons')
            elif self.conf.getEntry('srip_mode') == 'linear':
                self.srip_weight = self.conf.getEntry('srip_max')
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        
    def initModel(self, model: nn.Module, samples: Tensor) -> nn.Module:
        if self.conf.getEntry('model_init') == 'lsuv':
            assert samples is not None
            return LSUVinit(model, samples[torch.randperm(samples.shape[0])][: self.conf.getEntry('lsuv_size')], 
                            needed_mean=self.conf.getEntry('lsuv_mean'), needed_std=self.conf.getEntry('lsuv_std'), 
                            std_tol=self.conf.getEntry('lsuv_std_tol'), max_attempts=self.conf.getEntry('lsuv_maxiter'), 
                            do_orthonorm=self.conf.getEntry('lsuv_ortho'))
        return model
    
    
    def getOptimizer(self) -> optim.Optimizer:
        if self.conf.getEntry('optim_type') == 'sgd':
            if self.conf.getEntry('lr_mode') == 'fix':
                initial_lr = self.conf.getEntry('lr_cons')
            else:
                initial_lr = self.conf.getEntry('lr_max')
            if self.conf.getEntry('wd_mode') == 'fix':
                initial_wd = self.conf.getEntry('wd_cons')
            else:
                initial_wd = self.conf.getEntry('wd_min')
            momentum = self.conf.getEntry('momentum')
            return optim.SGD(self.model.parameters(), lr=initial_lr, momentum=momentum, weight_decay=initial_wd)
        raise ValueError('cannot obtain optimizer')
    
    
    def run(self) -> None:
        self.setup()
        
        self.epoch = 0
        self.validate()
        while self.epoch < self.epoch_max:
            self.adjust_lr()
            self.adjust_wd()
            if self.orth_regularizer == 'srip':
                self.adjust_srip()

            self.epoch += 1
            
            self.train()
            self.validate()
            
        torch.save(self.model.state_dict(), self.model_path)
        logging.info("Model saved successfully.")
            
            
    def train(self) -> None:
        logging.info(f'epoch: {self.epoch}, start training')
        
        recons_weight = self.conf.getEntry('reconstruct_weight')
        
        for one_batch, another_batch in zip(self.train_loader, self.train_query_loader):
            self.optimizer.zero_grad()
            
            one_batch = one_batch.to(self.device)
            another_batch = another_batch.to(self.device)
            
            one_embedding = self.model.encoder(one_batch)
            with torch.no_grad():
                another_batch = another_batch.detach()
                another_embedding = self.model.encoder(another_batch).detach()
            one_recons = self.model.decoder(one_embedding)
            
            trans_err = self.trans_loss_calculator(one_batch, another_batch, one_embedding, another_embedding)
            recons_err = recons_weight * self.recons_loss_calculator(one_batch, one_recons)
            orth_term = self.orth_reg().to(self.device)
            loss = trans_err + recons_err + orth_term
            
            loss.backward()
            self.optimizer.step()
            
            
    def validate(self) -> None:
        errors = []
        
        with torch.no_grad():
            for one_batch, another_batch in zip(self.val_loader, self.val_query_loader):
                one_batch = one_batch.to(self.device)  
                another_batch = another_batch.to(self.device)
                
                one_embedding = self.model.encoder(one_batch)
                another_embedding = self.model.encoder(another_batch)
                trans_err = self.trans_loss_calculator(one_batch, another_batch, one_embedding, another_embedding)

                errors.append(trans_err.detach().cpu())
                
        avg_error = torch.mean(torch.stack(errors)).item()
        logging.info(f'epoch: {self.epoch}, validate trans_err: {avg_error:.10f}')
    
    
    def adjust_lr(self) -> None:
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']
            break
        new_lr = current_lr
        if self.conf.getEntry('lr_mode') == 'linear':
            lr_max = self.conf.getEntry('lr_max')
            lr_min = self.conf.getEntry('lr_min')
            new_lr = lr_max - self.epoch * (lr_max - lr_min) / self.epoch_max
        elif self.conf.getEntry('lr_mode') == 'exponentiallyhalve':
            lr_max = self.conf.getEntry('lr_max')
            lr_min = self.conf.getEntry('lr_min')
            for i in range(1, 11):
                if (self.epoch_max - self.epoch) * (2 ** i) == self.epoch_max:
                    new_lr = lr_max / (10 ** i)
                    break
            if new_lr < lr_min:
                new_lr = lr_min
        elif self.conf.getEntry('lr_mode') == 'exponentially':
            lr_max = self.conf.getEntry('lr_max')
            lr_min = self.conf.getEntry('lr_min')
            lr_k = self.conf.getEntry('lr_everyk')
            lr_ebase = self.conf.getEntry('lr_ebase')
            lr_e = int(np.floor(self.epoch / lr_k))
            new_lr = lr_max * (lr_ebase ** lr_e)
            if new_lr < lr_min:
                new_lr = lr_min
        elif self.conf.getEntry('lr_mode') == 'plateauhalve':
            raise ValueError('plateauhalve is not yet supported')
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


    def adjust_wd(self):
        for param_group in self.optimizer.param_groups:
            current_wd = param_group['weight_decay']
            break
        new_wd = current_wd
        if self.conf.getEntry('wd_mode') == 'linear':
            wd_max = self.conf.getEntry('wd_max')
            wd_min = self.conf.getEntry('wd_min')
            new_wd = wd_min + self.epoch * (wd_max - wd_min) / self.epoch_max
        for param_group in self.optimizer.param_groups:
            param_group['weight_decay'] = new_wd
        
        
    def adjust_srip(self):
        if self.conf.getEntry('srip_mode') == 'linear':
            srip_max = self.conf.getEntry('srip_max')
            srip_min = self.conf.getEntry('srip_min')
            self.srip_weight = srip_max - self.epoch * (srip_max - srip_min) / self.epoch_max
    
    
    def orth_reg(self) -> torch.Tensor:
        if self.orth_regularizer == 'srip':
            return self.srip_weight * getSRIPTerm(self.model, self.device)
        return torch.zeros(1).to(self.device)
