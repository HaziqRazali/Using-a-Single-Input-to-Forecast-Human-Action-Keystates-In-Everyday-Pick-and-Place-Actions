import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist
 
def make_mlp(dim_list, activations, dropout=0):

    if len(dim_list) == 0 and len(activations) == 0:
        return nn.Identity()

    assert len(dim_list) == len(activations)+1
    
    layers = []
    for dim_in, dim_out, activation in zip(dim_list[:-1], dim_list[1:], activations):
                
        # append layer
        layers.append(nn.Linear(dim_in, dim_out))
        
        # # # # # # # # # # # # 
        # append activations  #
        # # # # # # # # # # # #
            
        activation_list = re.split('-', activation)
        for activation in activation_list:
                                        
            if 'leakyrelu' in activation:
                layers.append(nn.LeakyReLU(negative_slope=float(re.split('=', activation)[1]), inplace=True))
                
            elif activation == 'relu':
                layers.append(nn.ReLU())
                
            elif activation == "sigmoid":
                layers.append(nn.Sigmoid())
                
            elif activation == "none":
                pass
                                
            elif activation == "batchnorm":
                layers.append(nn.BatchNorm1d(dim_out))    
                        
            else:
                print("unknown activation")
                sys.exit()
            
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
                   
    return nn.Sequential(*layers)