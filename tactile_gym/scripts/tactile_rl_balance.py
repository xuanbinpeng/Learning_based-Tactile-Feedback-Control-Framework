from tactile_gym.rl_envs.demo_rl_env_base import demo_rl_env
from tactile_gym.rl_envs.nonprehensile_manipulation.object_balance.object_balance_env import (
    ObjectBalanceEnv,
)

import torch
import os
import numpy as np
from stable_baselines3 import DDPG, TD3, A2C,HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback
import h5py
import imageio
from sb3_contrib import TQC

def main():

    seed = int(0)
    num_iter = 10
    max_steps = 10000
    show_gui = False
    show_tactile = False
    render = False
    print_info = False
    model_test = False
    image_size = [128, 128]
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
   #env.action_space.np_random.seed(seed)
    print(env.observation_space)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    action_noise = NormalActionNoise(mean=np.zeros(env.action_space.shape[-1]), sigma=0.1 * np.ones(env.action_space.shape[-1]))
        
    policy_kwargs = dict(n_critics=2, activation_fn=torch.nn.ReLU, net_arch=[512, 256, 128])
    #policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[512, 256, 128])

    model = TQC("MultiInputPolicy", env,learning_rate=1e-3, batch_size=256,
               tau=0.001, gamma=0.95, action_noise=action_noise, buffer_size=int(1e6), train_freq=(5, 'step'),
               learning_starts=10000, policy_kwargs=policy_kwargs, device=device,tensorboard_log="../tensorboard_logs/balance/",verbose=1)
    #model = RecurrentPPO("MultiInputLstmPolicy", env, learning_rate=1e-3, batch_size=256, policy_kwargs=policy_kwargs, device=device,tensorboard_log="../tensorboard_logs/",verbose=1)
    render_frames = []
    #f = h5py.File('/home/nathan/tactile_gym/tactile_gym/vae/train_data/vae_data.h5', 'w')
    if model_test:
        #model = TQC.load("../model/tactile_vae.zip", env)
        obs = env.reset()
        sum_reward = 0
        for i in range(5):
            obs = env.reset()
            #print(obs)
            for i in range(1000):
                #print(type(obs))
                action,_ = model.predict(observation=obs)
                #print(type(action))
                #action = env.action_space.sample()
                #f.create_dataset('data_'+str(i), data=env.current_img)
                #print(action)
                obs, reward, done, info = env.step(action)
                #print(obs)
                sum_reward += reward
                if done:
                    break
                render_img = env.render()
                render_frames.append(render_img)
            print(sum_reward)
            sum_reward = 0        
            #imageio.mimwrite(os.path.join("example_videos", "balance.mp4"), np.stack(render_frames), fps=12)
    else :    
        if os.path.exists("../model/balance/best_model.zip"):
            #model = SAC.load("../model/tactile_vae.zip",env)
            print("pretrain")
        eval_callback = EvalCallback(env, best_model_save_path='../model_checkpoints/edge/',
                             log_path='../model_checkpoints/logs/', eval_freq=10000,
                             deterministic=True, render=False)
        model.learn(total_timesteps=500000, callback=eval_callback)
        model.save("../model_checkpoints/balance/model.zip")



if __name__ == "__main__":
    main()
