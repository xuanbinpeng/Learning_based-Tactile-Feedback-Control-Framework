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
import imageio
from sb3_contrib import TQC
import time

def main():

    seed = int(0)
    num_iter = 10
    max_steps = 1000
    show_gui = False
    model_test = False
    show_tactile = False
    render = False
    print_info = False
    image_size = [128, 128]


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
        #'observation_mode':'oracle',
        #"observation_mode": "tactile_and_feature",
        # 'observation_mode':'visual_and_feature',
        # 'observation_mode':'visuotactile_and_feature',
        # 'observation_mode':'encodedimg_and_feature',
        # 'observation_mode':'encodedimg',
        #'observation_mode':'privilege_feature',
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
    #env.action_space.np_random.seed(seed)
    #print(env.observation_space)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    action_noise = NormalActionNoise(mean=np.zeros(env.action_space.shape[-1]), sigma=0.1 * np.ones(env.action_space.shape[-1]))
        
    policy_kwargs = dict(n_critics=2, activation_fn=torch.nn.ReLU, net_arch=[512, 256, 128])
    #policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[512, 256, 128])

    model = TQC("MultiInputPolicy", env,top_quantiles_to_drop_per_net=2, learning_rate=1e-3, batch_size=256,
               tau=0.001, gamma=0.95, action_noise=action_noise, buffer_size=int(1e6), train_freq=(5, 'step'),
               learning_starts=10000, policy_kwargs=policy_kwargs, device=device,tensorboard_log="../tensorboard_logs/push/",verbose=1)
    #obs_list = np.load('./obs/obs.npy',allow_pickle=True)
    #model = RecurrentPPO("MultiInputLstmPolicy", env, learning_rate=1e-3, batch_size=256, policy_kwargs=policy_kwargs, device=device,tensorboard_log="../tensorboard_logs/",verbose=1)
    render_frames = []
    #f = h5py.File('/home/nathan/tactile_gym/tactile_gym/vae/train_data/vae_data.h5', 'w')
    if model_test:
        model = TQC.load("../model_checkpoints/push/best_model.zip", env)    #Q
        obs = env.reset()
        sum_reward = 0
        for i in range(10):
            obs = env.reset()
            #print(obs)
            for i in range(400):
                #print(obs["extended_feature"])
                time1 = time.time()
                action,_ = model.predict(observation=obs)
                time2 = time.time()
                print(time2-time1)
                #print(action)
                #print(type(action))
                #action = env.action_space.sample()
                #f.create_dataset('data_'+str(i), data=env.current_img

                obs, reward, done, info = env.step(action)
                #print(obs["extended_feature"][1])
                #print(obs_list[i])
                #obs = obs_list[i]
                print(obs)
                sum_reward += reward
                if done:
                    #print(i)
                    break
                render_img = env.render()
                render_frames.append(render_img)
            print(sum_reward)
            sum_reward = 0        
            #imageio.mimwrite(os.path.join("example_videos", "push.mp4"), np.stack(render_frames), fps=12)
    else :    
        if os.path.exists("../model_checkpoints/push/best_model.zip"):
            #model = SAC.load("../model/tactile_vae.zip",env)
            print("pretrain")
        #eval_callback = EvalCallback(env, best_model_save_path='../model_checkpoints/push/',
        #                     log_path='../model_checkpoints/logs/', eval_freq=10000,
        #                     deterministic=True, render=False)
        checkpoint_callback = CheckpointCallback(save_freq=100000, save_path='../model_checkpoints/push/')
        model.learn(total_timesteps=500000, callback=checkpoint_callback)
        model.save("../model_checkpoints/push/model.zip")
    
    

if __name__ == "__main__":
    main()
