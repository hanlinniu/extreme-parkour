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
        super().__init__()                                              # the depth_backbone/base_backbone is a DepthOnlyFCBackbone58x87 function. 
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
    
    # what is self.base_ang_vel?
    # what is priv_explicit?
    # what is heights?
    # how to apply get_heights function into the real robot?
    
    
# 4, check how to process the vision part?
  


# in line 89 of legged_robot_config.py, the depth class defined the simiulated camera position [0.27, 0, 0.03]
    


# all the info about observations are in legged_robot.py

# for setting the whole gym viewer position, check Line 548 of legged_robot.py:  def set_camera(self, position, lookat)

# for attaching camera to robot, check Line 945 of legged_robot.py:  def attach_camera(self, i, env_handle, actor_handle)

# change camera angle at Line 967 of legged_robot.py: camera_angle = np.random.uniform(config.angle[0], config.angle[1]), and angle is defined in Line 98 of legged_robot_config.py
    

# !!!!!!!!!!!!!!!!!!!!!
# for playing purpose, change camera angle at Line 100 of play_test_go2.py
# for train purpose, change angle defined in Line 98 of legged_robot_config.py

# in Line 114 of legged_robot.py: def step(self, actions):  
# remember that we need to do "actions = self.reindex(actions)" first
print("The final executed actions is: ", self.actions* self.cfg.control.action_scale)
print("Before execution, self.dof_pos is: ", self.dof_pos)
print("After execution, self.dof_pos is: ", self.dof_pos)
print("self.default_dof_pos_all is: ", self.default_dof_pos_all)
The final executed actions is:     tensor([[-0.0602, -0.2709,  0.1821, -0.0149,  0.0074,  0.0859, -0.0599, -0.0714, 0.1231, -0.0415, -0.4971, -0.1407]], device='cuda:0')
Before execution, self.dof_pos is: tensor([[ 0.0465,  0.3192, -1.4999, -0.1156,  0.8653, -1.4323,  0.0489,  0.8598, -1.4990, -0.1440,  0.6310, -1.7762]], device='cuda:0')
After execution, self.dof_pos is:  tensor([[ 0.0386,  0.3837, -1.4237, -0.1225,  0.8919, -1.4177,  0.0432,  0.8930, -1.4958, -0.1390,  0.5404, -1.7173]], device='cuda:0')
self.default_dof_pos_all is:       tensor([[ 0.1000,  0.8000, -1.5000, -0.1000,  0.8000, -1.5000,  0.1000,  1.0000, -1.5000, -0.1000,  1.0000, -1.5000]], device='cuda:0')

# Line 686 of legged_robot.py: def _compute_torques(self, actions):
actions_scaled = actions * self.cfg.control.action_scale
torques = self.p_gains*(actions_scaled + self.default_dof_pos_all - self.dof_pos) - self.d_gains*self.dof_vel

The 'self.root_states' can be found in Line 1034 of legged_robot.py:   
base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel  # base_init_state_list is a list with length 13

# The drawing function is in legged_robot.py



actor network is:  Actor(
  (priv_encoder): Sequential(
    (0): Linear(in_features=29, out_features=64, bias=True)
    (1): ELU(alpha=1.0)
    (2): Linear(in_features=64, out_features=20, bias=True)
    (3): ELU(alpha=1.0)
  )
  (history_encoder): StateHistoryEncoder(
    (activation_fn): ELU(alpha=1.0)
    (encoder): Sequential(
      (0): Linear(in_features=53, out_features=30, bias=True)
      (1): ELU(alpha=1.0)
    )
    (conv_layers): Sequential(
      (0): Conv1d(30, 20, kernel_size=(4,), stride=(2,))
      (1): ELU(alpha=1.0)
      (2): Conv1d(20, 10, kernel_size=(2,), stride=(1,))
      (3): ELU(alpha=1.0)
      (4): Flatten(start_dim=1, end_dim=-1)
    )
    (linear_output): Sequential(
      (0): Linear(in_features=30, out_features=20, bias=True)
      (1): ELU(alpha=1.0)
    )
  )
  (scan_encoder): Sequential(
    (0): Linear(in_features=132, out_features=128, bias=True)
    (1): ELU(alpha=1.0)
    (2): Linear(in_features=128, out_features=64, bias=True)
    (3): ELU(alpha=1.0)
    (4): Linear(in_features=64, out_features=32, bias=True)
    (5): Tanh()
  )
  (actor_backbone): Sequential(
    (0): Linear(in_features=114, out_features=512, bias=True)
    (1): ELU(alpha=1.0)
    (2): Linear(in_features=512, out_features=256, bias=True)
    (3): ELU(alpha=1.0)
    (4): Linear(in_features=256, out_features=128, bias=True)
    (5): ELU(alpha=1.0)
    (6): Linear(in_features=128, out_features=12, bias=True)
  )
)
critic network is:  Sequential(
  (0): Linear(in_features=753, out_features=512, bias=True)
  (1): ELU(alpha=1.0)
  (2): Linear(in_features=512, out_features=256, bias=True)
  (3): ELU(alpha=1.0)
  (4): Linear(in_features=256, out_features=128, bias=True)
  (5): ELU(alpha=1.0)
  (6): Linear(in_features=128, out_features=1, bias=True)
)


def compute_returns() is defined to calculate advantage function and returns function in rollout_storage.py, which will be later used in ppo.py's surrgate loss, value function loss, KL divergence, etc. 



In the real deployment, priv_explicit will be estimated by Line 148 of ppo.py
# priv_states_estimated = self.estimator(obs_est[:, :self.num_prop])


to train the estimator
check line 220 of ppo.py.
# self.estimator_optimizer.zero_grad() is in ppo.py


how to use estimator?
check line 142 of ppo.py


check the difference between these two in play_test_go2.py

    actions = ppo_runner.alg.depth_actor(obs_est.detach(), hist_encoding=True, scandots_latent=depth_latent)  # it is defined in actor, actor_critic.py
else:                                            # if there is no camera
    actions = policy(obs_est.detach(), hist_encoding=True, scandots_latent=depth_latent)


to understand self.root_states
# self.base_quat[:] = self.root_states[:, 3:7]
# self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
# self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])

# self.root_states[env_ids] = self.base_init_state
# base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel  # base_init_state_list is a list with length 13
# self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)

# class init_state:
#     pos = [0.0, 0.0, 1.] # x,y,z [m]
#     rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
#     lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
#     ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
#     default_joint_angles = { # target angles when action = 0.0
#         "joint_a": 0., 
#         "joint_b": 0.}


for the real action might be:
# actions_scaled = actions * self.cfg.control.action_scale    # action_scale = 0.5               in   def _compute_torques(self, actions):
# action_command = actions_scaled + self.default_dof_pos_all
# torques = self.motor_strength[0] * self.p_gains*(actions_scaled + self.default_dof_pos_all - self.dof_pos) - self.motor_strength[1] * self.d_gains*self.dof_vel 

for default joint angles is :
# class Go2RoughCfg( LeggedRobotCfg ):
# class init_state( LeggedRobotCfg.init_state ):
#     pos = [0.0, 0.0, 0.35] # x,y,z [m]
#     default_joint_angles = { # = target angles [rad] when action = 0.0
#         'FL_hip_joint': 0.1,   # [rad]
#         'RL_hip_joint': 0.1,   # [rad]
#         'FR_hip_joint': -0.1 ,  # [rad]
#         'RR_hip_joint': -0.1,   # [rad]

#         'FL_thigh_joint': 0.8,     # [rad]
#         'RL_thigh_joint': 1.,   # [rad]
#         'FR_thigh_joint': 0.8,     # [rad]
#         'RR_thigh_joint': 1.,   # [rad]

#         'FL_calf_joint': -1.5,   # [rad]
#         'RL_calf_joint': -1.5,    # [rad]
#         'FR_calf_joint': -1.5,  # [rad]
#         'RR_calf_joint': -1.5,    # [rad]


for last action
# self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)



obs_buf = torch.cat((self.base_ang_vel, self.obs_scales.ang_vel),dim=-1)



for contact_filt
# # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
# contact = self.contact_forces[:, self.feet_indices, 2] > 1.
# contact_filt = torch.logical_or(contact, self.last_contacts) 


tutorial for real robot
# https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/example/low_level/lowlevel_control.py
# https://support.unitree.com/home/en/developer/Python

