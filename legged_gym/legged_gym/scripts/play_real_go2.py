
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

def save_ppo_runner(ppo_runner, folder_path, filename="ppo_runner.pkl"):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    save_path = os.path.join(folder_path, filename)
    
    # Save the ppo_runner using pickle
    with open(save_path, 'wb') as f:
        pickle.dump(ppo_runner, f)
    
    print(f"PPO runner saved to {save_path}")

def load_ppo_runner(folder_path, filename="ppo_runner.pkl"):
    load_path = os.path.join(folder_path, filename)
    
    # Load the ppo_runner using pickle
    with open(load_path, 'rb') as f:
        ppo_runner = pickle.load(f)
    
    print(f"PPO runner loaded from {load_path}")
    return ppo_runner




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
    
    print("ppo_runner is: ", ppo_runner)
    save_folder = "~/extreme-parkour/legged_gym/legged_gym/scripts"


    if args.use_jit:
        path = os.path.join(log_pth, "traced")
        model, checkpoint = get_load_path(root=path, checkpoint=args.checkpoint)
        path = os.path.join(path, model)
        print("Loading jit for policy: ", path)
        policy_jit = torch.jit.load(path, map_location=env.device)
        print("it is loading base policy ##############################")
    else:
        policy = ppo_runner.get_inference_policy(device=env.device)             # this is the by default one
        print("it is loading base policy111111111 ##############################")

    estimator = ppo_runner.get_estimator_inference_policy(device=env.device)

    print("estimator is: ", estimator)

#     estimator is:  <bound method Estimator.inference of Estimator(
#   (estimator): Sequential(
#     (0): Linear(in_features=53, out_features=128, bias=True)
#     (1): ELU(alpha=1.0)
#     (2): Linear(in_features=128, out_features=64, bias=True)
#     (3): ELU(alpha=1.0)
#     (4): Linear(in_features=64, out_features=9, bias=True)
#   )
# )>