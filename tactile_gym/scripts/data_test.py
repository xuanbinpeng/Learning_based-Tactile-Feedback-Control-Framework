from tactile_gym.rl_envs.demo_rl_env_base import demo_rl_env
from tactile_gym.rl_envs.nonprehensile_manipulation.object_push.object_push_env import (
    ObjectPushEnv,
)
import torch
import os
import cv2
import numpy as np
from stable_baselines3 import DDPG, TD3, A2C,HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
import h5py
import imageio
from tactile_gym.vae_trans.model import VAE,FeatureMapping
from tactile_gym.Privileged_learning.model import Student
from queue import Queue
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = int(0)

max_steps = 1000
show_gui = False
show_tactile = False
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

Model_R = VAE
Model_S = VAE
Model_M = FeatureMapping

model_R = Model_R()
model_S = Model_S()
model_M = Model_M()

model_R.load_state_dict(torch.load("../tactile_gym/vae_trans/checkpoints/0_model.pt"))
model_S.load_state_dict(torch.load("../tactile_gym/vae_trans/checkpoints/1_model.pt"))
model_M.load_state_dict(torch.load("../tactile_gym/vae_trans/checkpoints/mapping_model.pt"))

model_R = model_R.to(device)
model_S = model_S.to(device)
model_M = model_M.to(device)

model = Student(use_gpu=True).to(device)
model.load_state_dict(torch.load("../tactile_gym/Privileged_learning/model/push/_4"))

real_path = '../tactile_gym/vae_trans/data/tactile_pair/tactile0/data_vae_pair.h5'
sim_path =  '../tactile_gym/vae_trans/data/tactile_pair/tactile1/data_vae_pair.h5'

def load_data(data_file_path):
    print("START LOADING H5")
    h5f_data = h5py.File(os.path.join(os.path.dirname(__file__), data_file_path), "r")


    print("len(h5f_data) = ",len(h5f_data))
    np_data = np.zeros((len(h5f_data), 128, 128), dtype=np.uint8)

    for i in range(len(h5f_data)):
        dset = h5f_data["data_"+str(i)]
        np_data[i] = np.asarray(dset[:])
        
    return np_data

if __name__ == "__main__":
    real = load_data(real_path)
    sim = load_data(sim_path)
    obs = {'extended_feature': np.array([0.000955, -0.000952, -0.000012, -0.000007, -0.000003, 0.008163,
       0.067500, 0.000000, 0.000000, 0.000000, 0.000000, 0.341720,
       0.097500, 0.010252, 0.000000, 0.000000, 0.000000, 0.387304]), 'privilege': np.array([0.009920, -0.009977, 0.000005, -0.000035, -0.000017, 0.085190,
       0.328374, 0.067932, 0.038424, -0.000204, -0.002153, -0.000001,
       0.000004, -0.086952, 0.009462, -0.002950, -0.007024, -0.000003,
       0.000018, 0.075617]), 'encodedimg': np.array([0.003074, 0.022847, 0.138618, -0.093046, 0.105447, 0.078645,
       -0.083670, 0.193847, 0.015014, -0.050942, -0.009687, 0.389902,
       0.047827, -0.122247, -0.048389, -0.000026, 0.125625, -0.037022,
       -0.052265, 0.150069, 0.111854, 0.043620, -0.088046, -0.062427,
       0.130656, -0.602013, 0.076786, 0.053831, -0.057261, 0.194531,
       -0.153059, 0.008735], dtype=np.float32)}
    
    action_fake = []
    action_real = []

    for i in range(800):
     
        #cv2.imshow("origin:",real[i])
        #cv2.imshow("gt:",sim[i])

        img_r = real[i] / 255
        img_r = img_r.reshape((1,1,128,128))
        img_r = torch.tensor(img_r, dtype=torch.float32).to(device)


        mu_r,_ = model_R.encoder(img_r)
        mu_s_fake = model_M(mu_r)


        obs["encodedimg"] = mu_s_fake.cpu().detach().numpy()[0]
        #print(obs["extended_feature"])
        obs_data = np.hstack((obs["extended_feature"], obs["encodedimg"]))
        obs_squence = Queue(maxsize=8) #队列
        for j in range(8):
            obs_squence.put(obs_data)   

        obs_batch = torch.tensor(obs_squence.queue).unsqueeze(0).float().to(device)

        action = model(obs_batch).cpu().detach().numpy()[0] * -1
        #print(action)
        #obs["encodedimg"] = mu_s_fake.cpu().detach().numpy()[0]

        
        obs_data = np.hstack((obs["extended_feature"], obs["encodedimg"]))
        obs_squence.get() #删除第一张图
        obs_squence.put(obs_data) #插入现在的图
    
        action_fake.append(action[0])

    for i in range(800):

        img_s = sim[i] / 255
        img_s = img_s.reshape((1,1,128,128))
        img_s = torch.tensor(img_s, dtype=torch.float32).to(device)

        mu_s_real,_ = model_S.encoder(img_s)


        obs["encodedimg"] = mu_s_real.cpu().detach().numpy()[0]
        obs_data = np.hstack((obs["extended_feature"], obs["encodedimg"]))
        obs_squence = Queue(maxsize=8) #队列
        for j in range(8):
            obs_squence.put(obs_data)   

        obs_batch = torch.tensor(obs_squence.queue).unsqueeze(0).float().to(device)
#  obs["encodedimg"] = mu_s_fake.cpu().detach().numpy()[0]

        action = model(obs_batch).cpu().detach().numpy()[0] * -1
        obs_data = np.hstack((obs["extended_feature"], obs["encodedimg"]))
        obs_squence.get() #删除第一张图
        obs_squence.put(obs_data) #插入现在的图
        #print(action)
        action_real.append(action[0])

    x = np.linspace(0, 800, 800)
 

    plt.plot(x, action_real, label='real')
    plt.plot(x, action_fake, label='fake')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Two Curves')

    plt.show()

        

        # recon_imgs = model_S.decoder(mu_sim).cpu().detach().numpy()

        # recon_imgs = recon_imgs.reshape((128,128))
        # recon_imgs = recon_imgs*255
        

