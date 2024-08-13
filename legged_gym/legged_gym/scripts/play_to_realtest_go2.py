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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import code

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from isaacgym import gymtorch, gymapi, gymutil
import numpy as np
import torch
import cv2
from collections import deque
import statistics
import faulthandler
from copy import deepcopy
import matplotlib.pyplot as plt
from time import time, sleep
from legged_gym.utils import webviewer
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import *
from rsl_rl.modules import Estimator

import pickle


def save_model(model, folder_path, filename):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    save_path = os.path.join(folder_path, filename)
    
    # Save the model using pickle
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"model saved to {save_path}")

def load_model(folder_path, filename):
    load_path = os.path.join(folder_path, filename)
    
    # Load the model using pickle
    with open(load_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"model loaded from {load_path}")
    return model


def save_step_data(tensor, folder_path, filename):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    save_path = os.path.join(folder_path, filename)
    
    # Save the tensor using torch.save
    torch.save(tensor, save_path)
    
    print(f"Data saved to {save_path}")

def load_step_data(folder_path, filename):
    load_path = os.path.join(folder_path, filename)
    
    # Load the depth tensor using torch.load
    tensor = torch.load(load_path)
    
    print(f"Data loaded from {load_path}")
    return tensor


def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    return model, checkpoint

def play(args):
    if args.web:
        web_viewer = webviewer.WebViewer()
    faulthandler.enable()
    exptid = args.exptid
    log_pth = "../../logs/{}/".format(args.proj_name) + args.exptid

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 0
    env_cfg.env.num_envs = 1 if not args.save else 64    #16 if not args.save else 64
    env_cfg.env.episode_length_s = 60
    env_cfg.commands.resampling_time = 60
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.height = [0.02, 0.02]
    env_cfg.terrain.terrain_dict = {"smooth slope": 0., 
                                    "rough slope up": 0.0,
                                    "rough slope down": 0.0,
                                    "rough stairs up": 0., 
                                    "rough stairs down": 0., 
                                    "discrete": 0., 
                                    "stepping stones": 0.0,
                                    "gaps": 0., 
                                    "smooth flat": 0,
                                    "pit": 0.0,
                                    "wall": 0.0,
                                    "platform": 0.,
                                    "large stairs up": 0.,
                                    "large stairs down": 0.,
                                    "parkour": 0.2,
                                    "parkour_hurdle": 0.2,
                                    "parkour_flat": 0.,
                                    "parkour_step": 0.2,
                                    "parkour_gap": 0.2, 
                                    "demo": 0.2}
    
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = True
    
    env_cfg.depth.angle = [20, 21]
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 6
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False

    depth_latent_buffer = []
    # prepare environment
    env: LeggedRobot
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    print("env is: ", env)

    obs = env.get_observations()

    if args.web:
        web_viewer.setup(env)

    ppo_runner: OnPolicyRunner
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg, return_log_dir=True)
    

    save_model_folder = os.path.expanduser("~/extreme-parkour/legged_gym/legged_gym/scripts/saved_models")
    save_data_folder = os.path.expanduser("~/extreme-parkour/legged_gym/legged_gym/scripts/saved_data")


    # if args.use_jit:
    #     path = os.path.join(log_pth, "traced")
    #     model, checkpoint = get_load_path(root=path, checkpoint=args.checkpoint)
    #     path = os.path.join(path, model)
    #     print("Loading jit for policy: ", path)
    #     policy_jit = torch.jit.load(path, map_location=env.device)
    #     print("it is loading base policy ##############################")
    # else:
    #     policy = ppo_runner.get_inference_policy(device=env.device)             # this is the by default one
    #     print("it is loading base policy111111111 ##############################")



    estimator = ppo_runner.get_estimator_inference_policy(device=env.device)
    # print("estimator is: ", estimator)
    # # save the estimator
    # save_model(estimator, folder_path=save_model_folder, filename="estimator.pkl")
    # Later, load the estimator
    # estimator = load_model(folder_path=save_model_folder, filename="estimator.pkl")
    print("Loaded estimator is: ", estimator)



    if env.cfg.depth.use_camera:
        depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)     # it is from on_policy_runner.py
    # save depth_encoder
    # save_model(depth_encoder, folder_path=save_model_folder, filename="depth_encoder.pkl")
    # Later, load the depth_encoder
    # depth_encoder = load_model(folder_path=save_model_folder, filename="depth_encoder.pkl")
    print("Loaded depth_encoder is: ", depth_encoder)


 
    depth_actor = ppo_runner.get_depth_actor_inference_policy(device=env.device)
    # # save depth_actor
    # save_model(depth_actor, folder_path=save_model_folder, filename="depth_actor.pkl")
    # Later, load the depth_actor
    # depth_actor = load_model(folder_path=save_model_folder, filename="depth_actor.pkl")
    print("Loaded depth_actor is: ", depth_actor)


    actions = torch.zeros(env.num_envs, 12, device=env.device, requires_grad=False)
    infos = {}
    # infos["depth"] = env.depth_buffer.clone().to(ppo_runner.device)[:, -1] if ppo_runner.if_depth else None
    infos["depth"] = env.depth_buffer.clone().to('cuda:0')[:, -1] if True else None
    # infos["depth"] size is  torch.Size([1, 58, 87])


    # infos["depth"] = load_step_data(folder_path=save_data_folder, filename="step_500_depth_data.pth")
    # obs = load_step_data(folder_path=save_data_folder, filename="step_500_obs_data.pth")
    # print('loaded depth data is ', infos["depth"])
    # print('loaded obs data is ', obs)


    # obs_student = obs[:, :53].clone()
    # obs_student[:, 6:8] = 0
    # ##################################################################################
    # # Input: infos[depth] (depth_image) size is  torch.Size([1, 58, 87]), obs_student (proprioception) size is  torch.Size([1, 53]) 
    # # Output: depth_latent_and_yaw size is :  torch.Size([1, 34])
    # depth_latent_and_yaw = depth_encoder(infos["depth"], obs_student)
    # ##################################################################################
    # # Input:  depth_latent_and_yaw size is :  torch.Size([1, 34])
    # # Output:  depth_latent size is :  torch.Size([1, 32]);  yaw size is : torch.Size([1, 2])
    # depth_latent = depth_latent_and_yaw[:, :-2]
    # yaw = depth_latent_and_yaw[:, -2:]
    # obs[:, 6:8] = 1.5*yaw

    # obs_est = obs.clone()
    # priv_states_estimated = estimator(obs_est[:, :53])         # output is 9
    # obs_est[:, 53+132:53+132+9] = priv_states_estimated
    # actions = ppo_runner.alg.depth_actor(obs_est.detach(), hist_encoding=True, scandots_latent=depth_latent)
    # # actions = depth_actor(obs_est.detach(), hist_encoding=True, scandots_latent=depth_latent)
    # print('Predicted action is: ', actions)



############################################# loop starts from here
    for i in range(1*int(env.max_episode_length)):
        print("#####################################################################")
        print("i is ", i)
        
        # if infos["depth"] is not None:
        #     print("infos[depth] is : ", infos["depth"])
        #     print("infos[depth] size is : ", infos["depth"].size())

        if i<100:
            step_depth_filename = f"step_{i}_depth_data.pth"
            step_obs_filename = f"step_{i}_obs_data.pth"
            save_step_data(infos["depth"], folder_path=save_data_folder, filename=step_depth_filename)
            save_step_data(obs, folder_path=save_data_folder, filename=step_obs_filename)
            print('saved depth data is ', infos["depth"])
            print('saved obs data is ', obs)

            # infos["depth"] = load_step_data(folder_path=save_data_folder, filename=step_depth_filename)
            # obs = load_step_data(folder_path=save_data_folder, filename=step_obs_filename)
            # print('loaded depth data is ', infos["depth"])
            # print('loaded obs data is ', obs)


            # obs_student = obs[:, :53].clone()
            # obs_student[:, 6:8] = 0
            # ##################################################################################
            # # Input: infos[depth] (depth_image) size is  torch.Size([1, 58, 87]), obs_student (proprioception) size is  torch.Size([1, 53]) 
            # # Output: depth_latent_and_yaw size is :  torch.Size([1, 34])
            # depth_latent_and_yaw = depth_encoder(infos["depth"], obs_student)
            # ##################################################################################
            # # Input:  depth_latent_and_yaw size is :  torch.Size([1, 34])
            # # Output:  depth_latent size is :  torch.Size([1, 32]);  yaw size is : torch.Size([1, 2])
            # depth_latent = depth_latent_and_yaw[:, :-2]
            # yaw = depth_latent_and_yaw[:, -2:]
            # obs[:, 6:8] = 1.5*yaw

            # obs_est = obs.clone()
            # priv_states_estimated = estimator(obs_est[:, :53])         # output is 9
            # obs_est[:, 53+132:53+132+9] = priv_states_estimated
            # actions = ppo_runner.alg.depth_actor(obs_est.detach(), hist_encoding=True, scandots_latent=depth_latent)
            # # actions = depth_actor(obs_est.detach(), hist_encoding=True, scandots_latent=depth_latent)
            # print(f'Predicted {i} action is: ', actions)

        if i==101:
            break

        if env.cfg.depth.use_camera:
            if infos["depth"] is not None:
                print("#####################################################################")
                print("it is using depth camera, infos has depth info")
                ##################################################################################
                # Input: obs size is :  torch.Size([1, 753]), env.cfg.env.n_proprio is :  53
                # Output: obs_student size is : torch.Size([1, 53])
                obs_student = obs[:, :env.cfg.env.n_proprio].clone()
                obs_student[:, 6:8] = 0
                ##################################################################################
                # Input: infos[depth] (depth_image) size is  torch.Size([1, 58, 87]), obs_student (proprioception) size is  torch.Size([1, 53]) 
                # Output: depth_latent_and_yaw size is :  torch.Size([1, 34])
                depth_latent_and_yaw = depth_encoder(infos["depth"], obs_student)
                ##################################################################################
                # Input:  depth_latent_and_yaw size is :  torch.Size([1, 34])
                # Output:  depth_latent size is :  torch.Size([1, 32]);  yaw size is : torch.Size([1, 2])
                depth_latent = depth_latent_and_yaw[:, :-2]
                yaw = depth_latent_and_yaw[:, -2:] # yaw is  tensor([[-0.0032,  0.2752]], device='cuda:0', grad_fn=<SliceBackward0>)
                # print("yaw is : ", yaw)
                ##################################################################################
            else:
                print("it is using depth camera, infos has no depth info")

            obs[:, 6:8] = 1.5*yaw
            
            
        else:
            print("it is not using depth camera")
            depth_latent = None
        
        # if hasattr(ppo_runner.alg, "depth_actor"):       # if there is 3D camera
        #     # Input: obs size is torch.Size([1, 753]), depth_latent size is :  torch.Size([1, 32])
        #     # Output: actions size is torch.Size([1, 12])
        #     actions = ppo_runner.alg.depth_actor(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)  # it is defined in actor, actor_critic.py
        # else:                                            # if there is no camera
        #     actions = policy(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)

        # try estimator here
        obs_est = obs.clone()
        priv_states_estimated = estimator(obs_est[:, :53])         # output is 9
        obs_est[:, 53+132:53+132+9] = priv_states_estimated

        # actions = depth_actor(obs_est.detach(), hist_encoding=True, scandots_latent=depth_latent)
        actions = ppo_runner.alg.depth_actor(obs_est.detach(), hist_encoding=True, scandots_latent=depth_latent)
        print(f'Predicted {i} action is: ', actions)

        # if hasattr(ppo_runner.alg, "depth_actor"):       # if there is 3D camera
        #     # Input: obs size is torch.Size([1, 753]), depth_latent size is :  torch.Size([1, 32])
        #     # Output: actions size is torch.Size([1, 12])
        #     # actions = ppo_runner.alg.depth_actor(obs_est.detach(), hist_encoding=True, scandots_latent=depth_latent)  # it is defined in actor of actor_critic.py
        #     actions = depth_actor(obs_est.detach(), hist_encoding=True, scandots_latent=depth_latent)
        #     print("it is using depth")
        # else:                                            # if there is no camera. it is running line 312 of actor_critic.py, also using actor of actor_critic.py
        #     # actions = policy(obs_est.detach(), hist_encoding=True, scandots_latent=depth_latent)   # if robot dog has camera, this line is never used. And depth_latent is always none
        #     print("it is not using depth")
        
            
        print("#####################################################################")
        obs, privileged_obs_buf, rews, dones, infos = env.step(actions.detach())   # infos is updated by legged_robot.py
        # print("infos[depth] is : ", infos["depth"])

        #######################################################################################
        # if args.web:
        #     web_viewer.render(fetch_results=True,
        #                 step_graphics=True,
        #                 render_all_camera_sensors=True,
        #                 wait_for_page_load=True)
        #######################################################################################
                
        print("time:", env.episode_length_buf[env.lookat_id].item() / 50, 
              "cmd vx", env.commands[env.lookat_id, 0].item(),
              "actual vx", env.base_lin_vel[env.lookat_id, 0].item(), )
        print("#####################################################################")
        
        id = env.lookat_id
        # print("look at id is : ", id)
        

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)