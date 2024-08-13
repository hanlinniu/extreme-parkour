
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

def play():
    save_folder = os.path.expanduser("~/extreme-parkour/legged_gym/legged_gym/scripts/saved_models")

    estimator = load_model(folder_path=save_folder, filename="estimator.pkl")
    print("Loaded estimator is: ", estimator)

    depth_encoder = load_model(folder_path=save_folder, filename="depth_encoder.pkl")
    print("Loaded depth_encoder is: ", depth_encoder)

    depth_actor = load_model(folder_path=save_folder, filename="depth_actor.pkl")
    print("Loaded depth_actor is: ", depth_actor)

    actions = torch.zeros((1), 12, device='cuda:0', requires_grad=False)
    print('actions is ', actions)

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    # args = get_args()
    play()
