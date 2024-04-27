import torch
import torch.nn as nn
import sys
import torchvision

class RecurrentDepthBackbone(nn.Module):
    def __init__(self, base_backbone, env_cfg) -> None:                 # this is used in depth_encoder = RecurrentDepthBackbone(depth_backbone, env.cfg).to(self.device) of on_policy_runner.py
        super().__init__()                                              # the depth_backbone is a DepthOnlyFCBackbone58x87 function. 
        activation = nn.ELU()
        last_activation = nn.Tanh()
        self.base_backbone = base_backbone
        if env_cfg == None:
            self.combination_mlp = nn.Sequential(
                                    nn.Linear(32 + 53, 128),
                                    activation,
                                    nn.Linear(128, 32)
                                )
        else:
            self.combination_mlp = nn.Sequential(
                                        nn.Linear(32 + env_cfg.env.n_proprio, 128),    # 32 + n_proprio(53) = 85
                                        activation,
                                        nn.Linear(128, 32)
                                    )
        self.rnn = nn.GRU(input_size=32, hidden_size=512, batch_first=True)
        self.output_mlp = nn.Sequential(
                                nn.Linear(512, 32+2),
                                last_activation
                            )
        self.hidden_states = None
        print("YESSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS in depth_backbone.py")
        print("base_backbone is : ", base_backbone)

    def forward(self, depth_image, proprioception):   # this function is used in "depth_latent_and_yaw = depth_encoder(infos["depth"], obs_student)" in play_test.py
        ######################################################################################################
        # Input: depth_image size is :  torch.Size([1, 58, 87])
        # Output: depth_image size is :  torch.Size([1, 32])
        depth_image = self.base_backbone(depth_image)       # this base_backbone used DepthOnlyFCBackbone58x87.forward function in depth_backbone.py

        ######################################################################################################
        # Input: depth_image size is :  torch.Size([1, 32]), proprioception size is :  torch.Size([1, 53])
        # Output: depth_latent size is :  torch.Size([1, 32])
        depth_latent = self.combination_mlp(torch.cat((depth_image, proprioception), dim=-1))    # -1 means concatenating these two tensors along the last dimension 
        
        # Input: depth_latent[:, None, :] size is :  torch.Size([1, 1, 32]. self.hidden_states size is : torch.Size([1, 1, 512]
        # Output: depth_latent size is :  torch.Size([1, 1, 512]), self.hidden_states size is :  torch.Size([1, 1, 512])
        depth_latent, self.hidden_states = self.rnn(depth_latent[:, None, :], self.hidden_states)
        # this is GRU (Gated Recurrent Unit) part, which needs a hidden states hat captures information from previous time steps.
        # The hidden state is updated at each time step, influenced by the current input and the previous hidden state.
        
        ######################################################################################################
        # Input: depth_latent size is :  torch.Size([1, 1, 512])
        # Output: depth_latent size is :  torch.Size([1, 34])
        depth_latent = self.output_mlp(depth_latent.squeeze(1))
        # print("depth_backbone.py is running")
   
        return depth_latent

    def detach_hidden_states(self):
        self.hidden_states = self.hidden_states.detach().clone()

class StackDepthEncoder(nn.Module):
    def __init__(self, base_backbone, env_cfg) -> None:
        super().__init__()
        activation = nn.ELU()
        self.base_backbone = base_backbone
        self.combination_mlp = nn.Sequential(
                                    nn.Linear(32 + env_cfg.env.n_proprio, 128),
                                    activation,
                                    nn.Linear(128, 32)
                                )

        self.conv1d = nn.Sequential(nn.Conv1d(in_channels=env_cfg.depth.buffer_len, out_channels=16, kernel_size=4, stride=2),  # (30 - 4) / 2 + 1 = 14,
                                    activation,
                                    nn.Conv1d(in_channels=16, out_channels=16, kernel_size=2), # 14-2+1 = 13,
                                    activation)
        self.mlp = nn.Sequential(nn.Linear(16*14, 32), 
                                 activation)
        
    def forward(self, depth_image, proprioception):
        # depth_image shape: [batch_size, num, 58, 87]
        depth_latent = self.base_backbone(None, depth_image.flatten(0, 1), None)  # [batch_size * num, 32]
        depth_latent = depth_latent.reshape(depth_image.shape[0], depth_image.shape[1], -1)  # [batch_size, num, 32]
        depth_latent = self.conv1d(depth_latent)
        depth_latent = self.mlp(depth_latent.flatten(1, 2))
        return depth_latent

    
class DepthOnlyFCBackbone58x87(nn.Module):
    def __init__(self, prop_dim, scandots_output_dim, hidden_state_dim, output_activation=None, num_frames=1):
        super().__init__()

        self.num_frames = num_frames
        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            # [1, 58, 87]
            nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=5),
            # [32, 54, 83]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [32, 27, 41]
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            activation,
            nn.Flatten(),
            # [32, 25, 39]
            nn.Linear(64 * 25 * 39, 128),
            activation,
            nn.Linear(128, scandots_output_dim)
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = activation

    def forward(self, images: torch.Tensor):
        images_compressed = self.image_compression(images.unsqueeze(1))
        latent = self.output_activation(images_compressed)

        return latent