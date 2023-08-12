from tactile_gym.rl_envs.demo_rl_env_base import demo_rl_env
from tactile_gym.rl_envs.exploration.surface_follow.surface_follow_vert.surface_follow_vert_env import (
    SurfaceFollowVertEnv,
)
import torch
import os
import numpy as np
from stable_baselines3 import DDPG, TD3, A2C,HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
import h5py
import imageio
from queue import Queue
from sb3_contrib import TQC
from tactile_gym.Privileged_learning.model import Student

def main():

    seed = int(0)
    num_iter = 10
    max_steps = 1000
    show_gui = True
    model_test = True
    show_tactile = False
    render = False
    print_info = False
    image_size = [128, 128]
    model_dir = './model/surface/'
    
    env_modes = {
        # which dofs can have movement
        # 'movement_mode':'yz',
        # 'movement_mode':'xyz',
        # 'movement_mode':'yzRx',
        "movement_mode": "xRz",

        # specify arm
        "arm_type": "ur5",

        # specify tactile sensor
        "tactile_sensor_name": "tactip",
        # "tactile_sensor_name": "digit",
        # "tactile_sensor_name": "digitac",

        # the type of control used
        # 'control_mode':'TCP_position_control',
        "control_mode": "TCP_velocity_control",

        # noise params for additional robustness
        "noise_mode": "vertical_simplex",

        # which observation type to return
        #'observation_mode': 'oracle',
        # "observation_mode": "tactile_and_feature",
        # 'observation_mode':'visual_and_feature',
        # 'observation_mode':'visuotactile_and_feature',
         'observation_mode':'encodedimg_privilege_feature',

        # which reward type to use (currently only dense)
        "reward_mode": "dense"
        # 'reward_mode':'sparse'
    }

    env = SurfaceFollowVertEnv(
        max_steps=max_steps,
        env_modes=env_modes,
        show_gui=show_gui,
        show_tactile=show_tactile,
        image_size=image_size,
    )

    # set seed for deterministic results
    env.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Student(use_gpu=True).to(device)
    model.load_state_dict(torch.load(model_dir+"_best"))
    obs = env.reset()
    sum_reward = 0

    for i in range(10):

        obs = env.reset()
        obs_data = np.hstack((obs["extended_feature"], obs["encodedimg"]))
        obs_squence = Queue(maxsize=8) #队列
        for j in range(8):
            obs_squence.put(obs_data)   
        for i in range(400):
            obs_batch = torch.tensor(obs_squence.queue).unsqueeze(0).float().to(device)

            action = model(obs_batch).cpu().detach().numpy()[0]
            #print(action)

            obs, reward, done, info = env.step(action)
            
            obs_data = np.hstack((obs["extended_feature"], obs["encodedimg"]))
            obs_squence.get() #删除第一张图
            obs_squence.put(obs_data) #插入现在的图

            sum_reward += reward
            if done:
                break
        print(sum_reward)
        sum_reward = 0    

 
            
 
    
    

if __name__ == "__main__":
    main()

