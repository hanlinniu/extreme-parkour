# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import code
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from torch.nn.modules.activation import ReLU


class StateHistoryEncoder(nn.Module):
    def __init__(self, activation_fn, input_size, tsteps, output_size, tanh_encoder_output=False):
        # self.device = device
        super(StateHistoryEncoder, self).__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps

        channel_size = 10
        # last_activation = nn.ELU()

        self.encoder = nn.Sequential(
                nn.Linear(input_size, 3 * channel_size), self.activation_fn,
                )

        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                    nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 8, stride = 4), self.activation_fn,
                    nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn,
                    nn.Conv1d(in_channels = channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn, nn.Flatten())
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 4, stride = 2), self.activation_fn,     # [3684, 20, 4]
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 2, stride = 1), self.activation_fn,         # [3684, 10, 3]
                nn.Flatten())                                                                                                                    # [3684, 30]
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 6, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Flatten())
        else:
            raise(ValueError("tsteps must be 10, 20 or 50"))

        self.linear_output = nn.Sequential(
                nn.Linear(channel_size * 3, output_size), self.activation_fn
                )

    def forward(self, obs):
        # nd * T * n_proprio
        nd = obs.shape[0]             # nd=36864
        T = self.tsteps               # 10
        # print("obs device", obs.device)
        # print("encoder device", next(self.encoder.parameters()).device)
        projection = self.encoder(obs.reshape([nd * T, -1])) # obs size is [36864, 10, 53];    obs.reshape([nd * T, -1]) size is [368640, 53];   projection size is [368640, 30]
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))  # projection.reshape([nd, T, -1]) size is [3684, 10, 30];  projection.reshape([nd, T, -1]).permute(0,2,1) size is [3684, 30, 10];   output size is [3684, 30]
        output = self.linear_output(output)   
        return output

class Actor(nn.Module):
    def __init__(self, num_prop, 
                 num_scan, 
                 num_actions, 
                 scan_encoder_dims,             # [128, 64, 32]        from legged_robot_config.py
                 actor_hidden_dims,             # [512, 256, 128]
                 priv_encoder_dims,             # [64, 20]
                 num_priv_latent,               # 29
                 num_priv_explicit,             # 9
                 num_hist, activation,          # 10 ELU(alpha=1.0)
                 tanh_encoder_output=False) -> None:
        super().__init__()
        # prop -> scan -> priv_explicit -> priv_latent -> hist
        # actor input: prop -> scan -> priv_explicit -> latent
        self.num_prop = num_prop     # 53
        self.num_scan = num_scan     # 132
        self.num_hist = num_hist     # 10
        self.num_actions = num_actions  # 12
        self.num_priv_latent = num_priv_latent    # 29
        self.num_priv_explicit = num_priv_explicit  # 9
        self.if_scan_encode = scan_encoder_dims is not None and num_scan > 0                      # true

        if len(priv_encoder_dims) > 0:          # [64, 20]             # priv_encoder: nn.linear           num_priv_latent is 29
                    priv_encoder_layers = []
                    priv_encoder_layers.append(nn.Linear(num_priv_latent, priv_encoder_dims[0]))
                    priv_encoder_layers.append(activation)
                    for l in range(len(priv_encoder_dims) - 1):
                        priv_encoder_layers.append(nn.Linear(priv_encoder_dims[l], priv_encoder_dims[l + 1]))
                        priv_encoder_layers.append(activation)
                    self.priv_encoder = nn.Sequential(*priv_encoder_layers)
                    priv_encoder_output_dim = priv_encoder_dims[-1]                 # 20
                    # print("############################################################")
                    # print("priv_encoder_output_dim is: ", priv_encoder_output_dim)
                    # print("priv_encoder_dims is: ", priv_encoder_dims)
                    # print("actor_hidden_dims is: ", actor_hidden_dims)
                    # print("num_priv_latent is: ", num_priv_latent)                # 29
                    # print("num_priv_explicit is: ", num_priv_explicit)
                    # print("num_hist is: ", num_hist)
                    # print("activation is: ", activation)
        else:
            self.priv_encoder = nn.Identity()
            priv_encoder_output_dim = num_priv_latent                # 29    Here it is a bit tricky

        print("it is depth branch")

        self.history_encoder = StateHistoryEncoder(activation, num_prop, num_hist, priv_encoder_output_dim)    # output is 20   # history_encoder:  nn.Conv1d
                                                               # 53      # 10 = history_len    # 20
        if self.if_scan_encode:              # True
            scan_encoder = []
            scan_encoder.append(nn.Linear(num_scan, scan_encoder_dims[0]))
            scan_encoder.append(activation)
            for l in range(len(scan_encoder_dims) - 1):                       # scan_encoder_dims is [128, 64, 32].  self.scan_encoder_output_dim is 32
                if l == len(scan_encoder_dims) - 2:
                    scan_encoder.append(nn.Linear(scan_encoder_dims[l], scan_encoder_dims[l+1]))
                    scan_encoder.append(nn.Tanh())
                else:
                    scan_encoder.append(nn.Linear(scan_encoder_dims[l], scan_encoder_dims[l + 1]))
                    scan_encoder.append(activation)
            self.scan_encoder = nn.Sequential(*scan_encoder)
            self.scan_encoder_output_dim = scan_encoder_dims[-1]               
        else:
            self.scan_encoder = nn.Identity()
            self.scan_encoder_output_dim = num_scan

        actor_layers = []
        actor_layers.append(nn.Linear(num_prop+                                # 53
                                      self.scan_encoder_output_dim+            # 32
                                      num_priv_explicit+                       # 9
                                      priv_encoder_output_dim,                 # 20
                                      actor_hidden_dims[0]))                   # 512 
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):                                # actor_hidden_dims is [512, 256, 128]
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))        # the last layer gives the output dimension of action, which is num_actions: 12
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        if tanh_encoder_output:
            actor_layers.append(nn.Tanh())
        self.actor_backbone = nn.Sequential(*actor_layers)

    def forward(self, obs, hist_encoding: bool, eval=False, scandots_latent=None):                    
        if not eval:                                                                      # eval can be False or True, both will work for play_test_go2.py
            # print("############################################################")
            # print(" it is not using eval")
            if self.if_scan_encode:              # True
                obs_scan = obs[:, self.num_prop:self.num_prop + self.num_scan]   # obs_scan dimension is 132
                if scandots_latent is None:                    
                    scan_latent = self.scan_encoder(obs_scan)   # if there is no vision, only simulated scandots.  actions_teacher is using this one with simulated scandots
                else:
                    scan_latent = scandots_latent               # if there is 3D camera, scandots_latent is not none     # 32
                obs_prop_scan = torch.cat([obs[:, :self.num_prop], scan_latent], dim=1)
            else:
                obs_prop_scan = obs[:, :self.num_prop + self.num_scan]
            obs_priv_explicit = obs[:, self.num_prop + self.num_scan:self.num_prop + self.num_scan + self.num_priv_explicit]     # obs_priv_explicit can be read from the robot directly
            if hist_encoding:                   # True
                latent = self.infer_hist_latent(obs)       # output is 20, infer privilege latent using history data
            else:
                latent = self.infer_priv_latent(obs)       # output is 20, input is 29, using privilege latent and priv_encoder directly, including mass_params_tensor, friction_coeffs_tensor, or motor_strength
            backbone_input = torch.cat([obs_prop_scan, obs_priv_explicit, latent], dim=1)        # length is 114 = 53 + 32    + 9(priv_explicit) + 20(latent, from priv_latent to smaller latent)
            backbone_output = self.actor_backbone(backbone_input)                                # length is 12
            return backbone_output
        else:
            # print("############################################################")
            # print(" it is using eval")
            if self.if_scan_encode:          
                obs_scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
                if scandots_latent is None:
                    scan_latent = self.scan_encoder(obs_scan)   
                else:
                    scan_latent = scandots_latent
                obs_prop_scan = torch.cat([obs[:, :self.num_prop], scan_latent], dim=1)
            else:
                obs_prop_scan = obs[:, :self.num_prop + self.num_scan]
            obs_priv_explicit = obs[:, self.num_prop + self.num_scan:self.num_prop + self.num_scan + self.num_priv_explicit]
            if hist_encoding:
                latent = self.infer_hist_latent(obs)
            else:
                latent = self.infer_priv_latent(obs)
            backbone_input = torch.cat([obs_prop_scan, obs_priv_explicit, latent], dim=1)
            backbone_output = self.actor_backbone(backbone_input)
            return backbone_output
    
    def infer_priv_latent(self, obs):
        priv = obs[:, self.num_prop + self.num_scan + self.num_priv_explicit: self.num_prop + self.num_scan + self.num_priv_explicit + self.num_priv_latent]
        return self.priv_encoder(priv)
    
    def infer_hist_latent(self, obs):
        hist = obs[:, -self.num_hist*self.num_prop:]
        return self.history_encoder(hist.view(-1, self.num_hist, self.num_prop))    #  hist.size size is [3684, 530];   hist.view(-1, self.num_hist, self.num_prop) size is [3684, 10, 53]
    
    def infer_scandots_latent(self, obs):
        scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
        return self.scan_encoder(scan)

class ActorCriticRMA(nn.Module):
    is_recurrent = False
    def __init__(self,  num_prop,
                        num_scan,
                        num_critic_obs,
                        num_priv_latent,               # 29
                        num_priv_explicit,             # 9
                        num_hist,
                        num_actions,
                        scan_encoder_dims=[256, 256, 256],                      # [128, 64, 32]
                        actor_hidden_dims=[256, 256, 256],                      # [512, 256, 128]
                        critic_hidden_dims=[256, 256, 256],                     # [512, 256, 128]
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticRMA, self).__init__()

        self.kwargs = kwargs
        priv_encoder_dims= kwargs['priv_encoder_dims']
        activation = get_activation(activation)
        
        self.actor = Actor(num_prop, num_scan, num_actions, scan_encoder_dims, actor_hidden_dims, priv_encoder_dims, num_priv_latent, num_priv_explicit, num_hist, activation, tanh_encoder_output=kwargs['tanh_encoder_output'])
        
        # print("############################################################")
        # print("scan_encoder_dims is : ", scan_encoder_dims)   # [128, 64, 32]
        # print("actor_hidden_dims is : ", actor_hidden_dims)   # [512, 256, 128]
        # print("critic_hidden_dims is : ", critic_hidden_dims) # [512, 256, 128]       

        # Value function                                           # [512, 256, 128]
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)
    
    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, hist_encoding):
        mean = self.actor(observations, hist_encoding)
        self.distribution = Normal(mean, mean*0. + self.std)    # self.std is a tensor with 12 variables, and it is dynamic

    def act(self, observations, hist_encoding=False, **kwargs):
        self.update_distribution(observations, hist_encoding)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):               # actions size is [6144, 12];   self.distribution.log_prob(actions) size is [6144, 12]
        return self.distribution.log_prob(actions).sum(dim=-1)             # it will sum all the 12 actions log_prob           self.distribution.log_prob(actions).sum(dim=-1) output size is [6144]

    def act_inference(self, observations, hist_encoding=False, eval=False, scandots_latent=None, **kwargs):
        if not eval:
            actions_mean = self.actor(observations, hist_encoding, eval, scandots_latent)   # during play_test_go2.py, it is using this line for non-camera scenario
            return actions_mean
        else:
            actions_mean, latent_hist, latent_priv = self.actor(observations, hist_encoding, eval=True)
            return actions_mean, latent_hist, latent_priv

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
    
    def reset_std(self, std, num_actions, device):
        new_std = std * torch.ones(num_actions, device=device)
        self.std.data = new_std.data

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
