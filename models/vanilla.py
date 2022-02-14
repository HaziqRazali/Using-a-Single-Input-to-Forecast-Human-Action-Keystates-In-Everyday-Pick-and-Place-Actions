import torch.distributions as tdist
import torch.nn.functional as F
import torch
import torch.nn as nn

from torch.nn import GRU, LSTM
import time

from models.components import *

class model(nn.Module):
    def __init__(self, args):
        super(model, self).__init__()
                 
        for key, value in args.__dict__.items():
            setattr(self, key, value)
                        
        """
        Pose
        """
          
        self.inp_pose_encoder = make_mlp([self.pose_encoder_units[0]+3]+self.pose_encoder_units[1:],self.pose_encoder_activations)
        self.key_pose_encoder = make_mlp(self.pose_encoder_units,self.pose_encoder_activations)
        self.pose_mu          = make_mlp(self.pose_mu_var_units,self.pose_mu_var_activations)
        self.pose_log_var     = make_mlp(self.pose_mu_var_units,self.pose_mu_var_activations)
        self.key_pose_decoder = make_mlp(self.pose_decoder_units,self.pose_decoder_activations)
        
        """
        Time
        """
        
        self.delta_pose_encoder = make_mlp(self.delta_pose_encoder_units,self.delta_pose_encoder_activations)
        self.time_encoder       = make_mlp(self.time_encoder_units,self.time_encoder_activations)
        self.time_mu            = make_mlp(self.time_mu_var_units,self.time_mu_var_activations)
        self.time_log_var       = make_mlp(self.time_mu_var_units,self.time_mu_var_activations)
        self.time_decoder       = make_mlp(self.time_decoder_units,self.time_decoder_activations)
                
        self.norm = tdist.Normal(torch.tensor(0.0), torch.tensor(1.0))
                        
    def forward(self, data, mode):
    
        inp_pose = data["inp_pose"]      #.view(self.batch_size,-1)
        inp_center = data["inp_center"]  #.view(self.batch_size,-1)
        key_pose = data["key_pose"]      #.view(self.batch_size,-1)
        key_center = data["key_center"]  #.view(self.batch_size,-1)
        key_object = data["key_object"]  #.view(self.batch_size,-1) # [batch, 1, 3]
        approach_angle = data["approach_angle"] #.view(self.batch_size,-1)
        total_distance = data["total_distance"] 
                
        # set height of key object to zero so it only encodes direction
        key_object[:,0,-1] = 0
                        
        """
        compute pose
        """
    
        # feed x and y
        inp_pose_features = torch.cat((inp_pose.view(self.batch_size,-1), approach_angle), dim=1)
        inp_pose_features = self.inp_pose_encoder(inp_pose_features)
        key_pose_features = self.key_pose_encoder((key_pose - key_object).view(self.batch_size,-1))
        
        # get gaussian parameters
        pose_posterior = torch.cat((inp_pose_features,key_pose_features),dim=1)
        pose_posterior_mu = self.pose_mu(pose_posterior)
        pose_posterior_log_var = self.pose_log_var(pose_posterior)
        
        # sample
        pose_posterior_std = torch.exp(0.5*pose_posterior_log_var)
        pose_posterior_eps = self.norm.sample([self.batch_size, pose_posterior_mu.shape[1]]).cuda()
        pose_posterior_z   = pose_posterior_mu + pose_posterior_eps*pose_posterior_std
                
        z_p = torch.unsqueeze(pose_posterior_z,0).repeat(self.num_samples,1,1) if mode == "tr" else self.norm.sample([self.num_samples, self.batch_size, self.pose_mu_var_units[-1]]).cuda()
        
        # forecast
        pred_key_pose = torch.cat((z_p,torch.unsqueeze(inp_pose_features,0).repeat(self.num_samples,1,1)),dim=-1)
        pred_key_pose = self.key_pose_decoder(pred_key_pose)
        pred_key_pose = pred_key_pose.view(self.num_samples,self.batch_size,21,3) + torch.unsqueeze(key_object,0)
        
        """
        compute time
        """
                
        centered_key_pose = pred_key_pose - pred_key_pose[:,:,11:12,:]
        centered_key_pose[:,:,:,:2] = centered_key_pose[:,:,:,:2] + torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(total_distance,1),1),0)
                
        # compute delta_pose
        delta_pose = centered_key_pose - torch.unsqueeze(inp_pose,0)
        delta_pose = delta_pose.view(self.num_samples,self.batch_size,-1)
        
        # feed x and y
        delta_pose_features = self.delta_pose_encoder(delta_pose)
        time_features = self.time_encoder(data["time"].unsqueeze(1))
                
        # get gaussian parameters
        time_posterior = torch.cat((delta_pose_features,torch.unsqueeze(time_features,0).repeat(self.num_samples,1,1)),dim=-1)
        time_posterior_mu = self.time_mu(time_posterior)
        time_posterior_log_var = self.time_log_var(time_posterior)
                
        # sample
        time_posterior_std = torch.exp(0.5*time_posterior_log_var)
        time_posterior_eps = self.norm.sample([self.num_samples, self.batch_size, time_posterior_mu.shape[-1]]).cuda()
        time_posterior_z   = time_posterior_mu + time_posterior_eps*time_posterior_std
        
        z_t = time_posterior_z if mode == "tr" else self.norm.sample([self.num_samples,self.batch_size, self.time_mu_var_units[-1]]).cuda()
        
        # compute time
        time = torch.cat((z_t,delta_pose_features),dim=-1)
        time = torch.squeeze(self.time_decoder(time),dim=-1)
                                
        return {"key_pose":torch.mean(pred_key_pose,dim=0), "time":torch.mean(time,dim=0),
                "pose_posterior":{"mu":pose_posterior_mu, "log_var":pose_posterior_log_var}, 
                "time_posterior":{"mu":time_posterior_mu, "log_var":time_posterior_log_var},
                "pred_key_poses":pred_key_pose.permute(1,0,2,3),"times":time.permute(1,0)}