from tactile_gym.rl_envs.demo_rl_env_base import demo_rl_env
from tactile_gym.rl_envs.nonprehensile_manipulation.object_push.object_push_env import (
    ObjectPushEnv,
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
import time
from queue import Queue
from sb3_contrib import TQC
from tactile_gym.Privileged_learning.model import Student

def main():

    seed = int(0)
    num_iter = 10
    max_steps = 1000
    show_gui = True
    model_test = True
    show_tactile = True
    render = False
    print_info = False
    image_size = [128, 128]
    model_dir = './model/push/'
    
    env_modes = {
        # which dofs can have movement (environment dependent)
        # 'movement_mode':'y',
        # 'movement_mode':'yRz',
        #"movement_mode": "xyRz",
        'movement_mode': 'TyRz',
        #'movement_mode':'TxTyRz',

        # specify arm
        "arm_type": "ur5",
        # "arm_type": "mg400",

        # specify tactile sensor
        "tactile_sensor_name": "tactip",
        # "tactile_sensor_name": "digit",
        # "tactile_sensor_name": "digitac",

        # the type of control used 
        #'control_mode':'TCP_position_control',
        "control_mode": "TCP_velocity_control",

        # randomisations
        "rand_init_orn": True,
        "rand_obj_mass": True,
        "fix_obj":False,

        # straight or random trajectory
        # "traj_type": "straight",
        'traj_type': 'simplex',

        # which observation type to return
        # 'observation_mode':'oracle',
        #"observation_mode": "tactile_and_feature",
        # 'observation_mode':'visual_and_feature',
        # 'observation_mode':'visuotactile_and_feature',
        # 'observation_mode':'encodedimg_and_feature',
        # 'observation_mode':'encodedimg',
        'observation_mode':'encodedimg_privilege_feature',

        # the reward type
        "reward_mode": "dense"
        # 'reward_mode':'sparse'
    }

    env = ObjectPushEnv(
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

    for i in range(1):

        obs = env.reset()
        obs_data = np.hstack((obs["extended_feature"], obs["encodedimg"]))
        obs_squence = Queue(maxsize=8) #队列
        for j in range(8):
            obs_squence.put(obs_data)   
        for i in range(50):
            obs_batch = torch.tensor(obs_squence.queue).unsqueeze(0).float().to(device)
            time1 = time.time()
            # action = model(obs_batch).cpu().detach().numpy()[0]
            action = [0.1,0.1]
            time2 = time.time()
            #print(time2-time1)
            # print(action)

            obs, reward, done, info = env.step(action)
            print(obs["extended_feature"][0:6])
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

