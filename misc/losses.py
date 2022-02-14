import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.functional import cross_entropy

class compute_loss(nn.Module):
    def __init__(self, args):
        super(compute_loss, self).__init__()
    
        for key, value in args.__dict__.items():
            setattr(self, key, value)

    # # # # # # # # #
    # forward pass  #
    # # # # # # # # #

    def forward(self, inp_data, out_data, loss_name, loss_function):
    
        # kl divergence
        if any(x == loss_function for x in ["kl_loss"]):
            loss = eval("self."+loss_function)(out_data[loss_name])
        
        # l1, mse, soft_cross_entropy
        elif any(x == loss_function for x in ["mse"]):
            loss = eval("self."+loss_function)(out_data[loss_name], inp_data[loss_name])
        
        # error
        else:
            print("Unknown loss function:", loss_function)
            sys.exit()
            
        # make sure the final loss is average over the batch size
        losses = torch.sum(loss) / loss.shape[0] if len(loss.shape) != 0 else loss        
        return losses

    # # # # # # # # # #
    # loss functions  #
    # # # # # # # # # #   
        
    # mse loss
    ####################################################
    def mse(self, pred_data, true_data):

        # pred_data.shape = [batch, M, N, ...]
        # true_data.shape = [batch, M, N, ...]
        batch = pred_data.shape[0]
        pred_data = pred_data.view(batch,-1)
        true_data = true_data.view(batch,-1)
        
        loss = torch.sum((true_data - pred_data)**2,dim=1) # [32]
        return torch.sum(loss) / loss.shape[0]

    # KL Divergence of mu and sigma
    ####################################################    
    def kl_loss(self, out_data):
    
        # inp_data not used
        
        kl_loss = -0.5 * (out_data["log_var"] - out_data["log_var"].exp() - out_data["mu"].pow(2) +1) # [batch, num_units]
        kl_loss = torch.sum(kl_loss, dim=1) # [batch]
        return torch.sum(kl_loss) / kl_loss.shape[0]
        