########################################################################################################################
# in play_test.py
depth_latent_and_yaw = depth_encoder(infos["depth"], obs_student)
depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)
ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg, return_log_dir=True)

########################################################################################################################
# in Task_registry.py
runner = OnPolicyRunner(env, 
                                train_cfg_dict, 
                                log_dir, 
                                init_wandb=init_wandb,
                                device=args.rl_device, **kwargs)

# in on_policy_runner.py
if self.if_depth:
    depth_backbone = DepthOnlyFCBackbone58x87(env.cfg.env.n_proprio, 
                                            self.policy_cfg["scan_encoder_dims"][-1], 
                                            self.depth_encoder_cfg["hidden_dims"],
                                            )
    depth_encoder = RecurrentDepthBackbone(depth_backbone, env.cfg).to(self.device)
    depth_actor = deepcopy(actor_critic.actor) # it is defined in actor_critic.py line 232 and line 159, self.actor = Actor(num_prop, num_scan, num_actions, scan_encoder_dims, actor_hidden_dims, priv_encoder_dims, num_priv_latent, num_priv_explicit, num_hist, activation, tanh_encoder_output=kwargs['tanh_encoder_output'])
else:
    depth_encoder = None
    depth_actor = None

########################################################################################################################
# in depth_backbone.py

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
                                        nn.Linear(32 + env_cfg.env.n_proprio, 128),
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
        print("Before depth_backbone.py, depth_image size is : ", depth_image.size()) #Before depth_backbone.py, depth_image size is :  torch.Size([1, 58, 87])
        depth_image = self.base_backbone(depth_image)       # this base_backbone used DepthOnlyFCBackbone58x87.forward function in depth_backbone.py
        print("After depth_backbone.py, depth_image size is : ", depth_image.size()) #After depth_backbone.py, depth_image size is :  torch.Size([1, 32])
        print("proprioception size is : ", proprioception.size())
        depth_latent = self.combination_mlp(torch.cat((depth_image, proprioception), dim=-1))
        # depth_latent = self.base_backbone(depth_image)
        depth_latent, self.hidden_states = self.rnn(depth_latent[:, None, :], self.hidden_states)
        depth_latent = self.output_mlp(depth_latent.squeeze(1))
        print("Forwardddddddddddddddddddddddddddddddddddddddddddddddddddd in depth_backbone.py")
        print("In depth_backbone.py, proprioception size is : ", proprioception.size()) #In depth_backbone.py, proprioception size is :  torch.Size([1, 53])
        print("After output_mlp, depth_latent size is : ", depth_latent.size())
        
        return depth_latent

    def detach_hidden_states(self):
        self.hidden_states = self.hidden_states.detach().clone()

    
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
    



# remaining questions:
# 1, understand line 466 of legged_robot.py
# self.obs_buf = torch.cat([obs_buf,    heights,            priv_explicit,   priv_latent,        self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
                #          [1, 53],     [1, 132] n_scan,    [1, 9],          [1, 29] constant,   [1, 530]  53*10                          # total is [1, 753]
            

# 2, what is heights,  priv_explicit,   priv_latent?
    # priv_explicit size is:  tensor([[1.1641, 0.2642, 0.2127, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]
    # priv_latent is constant
    # what is base_lin_vel?
# 3, what is obs_buf consisting of?
    
    
# 4, check how to process the vision part?
    

# 5, go to manchester or oxford, and test
    


# all the info about observations are in legged_robot.py
    

# how to apply get_heights function into the real robot?



