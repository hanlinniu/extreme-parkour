from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import code


import numpy as np
import torch
import cv2
from collections import deque
import statistics
import faulthandler
from copy import deepcopy
import matplotlib.pyplot as plt
from time import time, sleep

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import *
from rsl_rl.modules import Estimator
import time

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


def play():
    save_folder = os.path.expanduser("~/extreme-parkour/legged_gym/legged_gym/scripts/saved_models")
    save_data_folder = os.path.expanduser("~/extreme-parkour/legged_gym/legged_gym/scripts/saved_data")

    estimator = load_model(folder_path=save_folder, filename="estimator.pkl")
    print("Loaded estimator is: ", estimator)

    depth_encoder = load_model(folder_path=save_folder, filename="depth_encoder.pkl")
    print("Loaded depth_encoder is: ", depth_encoder)

    depth_actor = load_model(folder_path=save_folder, filename="depth_actor.pkl")
    print("Loaded depth_actor is: ", depth_actor)

    actions = torch.zeros((1), 12, device='cuda:0', requires_grad=False)
    print('actions is ', actions)

    infos = {}

    for i in range(102):
        step_depth_filename = f"step_{i}_depth_data.pth"
        step_obs_filename = f"step_{i}_obs_data.pth"
        infos["depth"] = load_step_data(folder_path=save_data_folder, filename=step_depth_filename)
        obs = load_step_data(folder_path=save_data_folder, filename=step_obs_filename)
        print('loaded depth data is ', infos["depth"])
        print('loaded obs data is ', obs)

        if infos["depth"] is not None:
            obs_student = obs[:, :53].clone()
            obs_student[:, 6:8] = 0
            ##################################################################################
            # Input: infos[depth] (depth_image) size is  torch.Size([1, 58, 87]), obs_student (proprioception) size is  torch.Size([1, 53]) 
            # Output: depth_latent_and_yaw size is :  torch.Size([1, 34])
            depth_latent_and_yaw = depth_encoder(infos["depth"], obs_student)
            ##################################################################################
            # Input:  depth_latent_and_yaw size is :  torch.Size([1, 34])
            # Output:  depth_latent size is :  torch.Size([1, 32]);  yaw size is : torch.Size([1, 2])
            depth_latent = depth_latent_and_yaw[:, :-2]
            yaw = depth_latent_and_yaw[:, -2:]
        else:
            print("it is using depth camera, infos has no depth info")
            
        obs[:, 6:8] = 1.5*yaw

        obs_est = obs.clone()
        priv_states_estimated = estimator(obs_est[:, :53])         # output is 9
        obs_est[:, 53+132:53+132+9] = priv_states_estimated
        # actions = ppo_runner.alg.depth_actor(obs_est.detach(), hist_encoding=True, scandots_latent=depth_latent)
        actions = depth_actor(obs_est.detach(), hist_encoding=True, scandots_latent=depth_latent)
        print(f'Predicted {i} action is: ', actions)
        time.sleep(0.3)
        if i==101:
            break
    
if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    # args = get_args()
    play()
