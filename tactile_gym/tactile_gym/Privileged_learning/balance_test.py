from tactile_gym.rl_envs.demo_rl_env_base import demo_rl_env
from tactile_gym.rl_envs.nonprehensile_manipulation.object_balance.object_balance_env import (
    ObjectBalanceEnv,
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
    model_dir = './model/balance/'
    
    env_modes = {
        # which dofs can have movement (environment dependent)
        "movement_mode": "xy",
        # 'movement_mode':'xyz',
        # 'movement_mode':'RxRy',
        # 'movement_mode':'xyRxRy',

        # specify arm
        "arm_type": "ur5",

        # specify tactile sensor
        "tactile_sensor_name": "tactip",
        # "tactile_sensor_name": "digit",
        # "tactile_sensor_name": "digitac",


        # whether to use spin plate on initialisation
        "object_mode": "pole",
        # 'object_mode':'ball_on_plate',
        # 'object_mode':'spinning_plate',

        # the type of control used
        # 'control_mode':'TCP_position_control',
        "control_mode": "TCP_velocity_control",

        # add variation to joint force for rigid core
        "rand_gravity": False,
        "rand_embed_dist": True,

        # which observation type to return
        #'observation_mode': 'oracle',
        # "observation_mode": "tactile",
        # 'observation_mode':'visual',
        # 'observation_mode':'visuotactile',
        # 'observation_mode':'encodedimg_and_feature',
        # 'observation_mode':'encodedimg',
         'observation_mode':'encodedimg_privilege_feature',

        # the reward type
        "reward_mode": "dense"
        # 'reward_mode':'sparse'
    }

    env = ObjectBalanceEnv(
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

