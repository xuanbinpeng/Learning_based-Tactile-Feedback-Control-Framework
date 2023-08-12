from tactile_gym.rl_envs.demo_rl_env_base import demo_rl_env
from tactile_gym.rl_envs.exploration.surface_follow.surface_follow_goal.surface_follow_goal_env import (
    SurfaceFollowGoalEnv,
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
from sb3_contrib import TQC

def main():

    seed = int(0)
    num_iter = 100
    max_steps = 1000
    show_gui = True
    show_tactile = False
    render = False
    model_test = True
    print_info = False
    image_size = [128, 128]
    env_modes = {
        # which dofs can have movement
        # 'movement_mode':'yz',
        # 'movement_mode':'xyz',
        # 'movement_mode':'yzRx',
        #"movement_mode": "xRz",
        "movement_mode": "xyzRxRy",

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
        "noise_mode": "simplex",

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

    env = SurfaceFollowGoalEnv(
        max_steps=max_steps,
        env_modes=env_modes,
        show_gui=show_gui,
        show_tactile=show_tactile,
        image_size=image_size,
    )

    # set seeding
    env.seed(seed)
    
   #env.action_space.np_random.seed(seed)
    print(env.observation_space)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    action_noise = NormalActionNoise(mean=np.zeros(env.action_space.shape[-1]), sigma=0.1 * np.ones(env.action_space.shape[-1]))
        
    policy_kwargs = dict(n_critics=2, activation_fn=torch.nn.ReLU, net_arch=[512, 256, 128])
    #policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[512, 256, 128])

    model = TQC("MultiInputPolicy", env,learning_rate=1e-3, batch_size=256,
               tau=0.001, gamma=0.95, action_noise=action_noise, buffer_size=int(1e6), train_freq=(5, 'step'),
               learning_starts=10000, policy_kwargs=policy_kwargs, device=device,tensorboard_log="../tensorboard_logs/surface/",verbose=1)
    #model = RecurrentPPO("MultiInputLstmPolicy", env, learning_rate=1e-3, batch_size=256, policy_kwargs=policy_kwargs, device=device,tensorboard_log="../tensorboard_logs/",verbose=1)
    render_frames = []
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
                
                action = [0,0,0,0,0.1]
                print(action)
                obs, reward, done, info = env.step(action)
                #print(obs)
                sum_reward += reward
                if done:
                    break
                render_img = env.render()
                render_frames.append(render_img)
            #print(sum_reward)
            sum_reward = 0        
            #imageio.mimwrite(os.path.join("example_videos", "surface.mp4"), np.stack(render_frames), fps=12)
    else :    
        if os.path.exists("../model/surface/best_model.zip"):
            #model = SAC.load("../model/tactile_vae.zip",env)
            print("pretrain")
        eval_callback = EvalCallback(env, best_model_save_path='../model_checkpoints/edge/',
                             log_path='../model_checkpoints/logs/', eval_freq=10000,
                             deterministic=True, render=False)
        model.learn(total_timesteps=500000, callback=eval_callback)
        model.save("../model_checkpoints/surface/model.zip")
    
    

if __name__ == "__main__":
    main()
