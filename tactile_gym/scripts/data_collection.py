import time
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import torch
import numpy as np
import h5py
import random
from sb3_contrib import TQC
from stable_baselines3.common.noise import NormalActionNoise
from tactile_gym.rl_envs.nonprehensile_manipulation.object_balance.object_balance_env import (
    ObjectBalanceEnv,
)
from tactile_gym.rl_envs.nonprehensile_manipulation.object_push.object_push_env import (
    ObjectPushEnv,
)
seed = int(0)
num_iter = 10
max_steps = 1600
show_gui = True
model_test = True
show_tactile = True
render = False
print_info = False
image_size = [128, 128]
env_modes = {
        # which dofs can have movement (environment dependent)
        # 'movement_mode':'y',
        # 'movement_mode':'yRz',
        #"movement_mode": "xyRz",
        #'movement_mode': 'TyRz',
        #'movement_mode':'TxTyRz',
         'movement_mode':'TxRyRz',

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
        "rand_obj_mass": False,
        "fix_obj":True,

        # straight or random trajectory
        # "traj_type": "straight",
        'traj_type': 'simplex',

        # which observation type to return
        # 'observation_mode':'oracle',
        #"observation_mode": "tactile_and_feature",
        # 'observation_mode':'visual_and_feature',
        # 'observation_mode':'visuotactile_and_feature',
         'observation_mode':'encodedimg_and_feature',
        # 'observation_mode':'encodedimg',

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

v10=0
v11=0.2
v12=0.2
v13=0.2

v20=0.25
v21=0
v22=0.2
v23=-0.2

# control depth   1100                    #0.08 0.12 0.16 0.20 (unpaired)    0.12 (paired)
action2x0 = np.ones(5)*(0.08)
action2x1 = np.zeros(340)

action2x0_ = np.ones(10)*(0.02)
action2x1_ = np.zeros(240)

action2x0__ = np.ones(10)*(0.03)
action2x1__ = np.zeros(200)





action2x2 = np.ones(10)*(0.015)
action2x3 = np.zeros(200)

action2x4 = np.ones(10)*(0.015)
action2x5 = np.zeros(200)

action2x6 = np.ones(10)*(0)
action2x7 = np.zeros(200)

action2x8 = np.ones(10)*(0.025)
action2x9 = np.ones(15)*(0.085)
action2x10 = np.ones(80)*(-0.030)
action2x11 = np.ones(60)*(0)



# action2x11 = np.ones(100)*0
# action2x12 = np.ones(10)*(0.025)
# action2x13 = np.ones(10)*(0.1)
# action2x14 = np.ones(40)*(0.1)*0



#np.random.seed()
action2y1 = np.zeros(1000)
#action2y1=np.random.random(1045) * (np.random.random(1045)-0.5)
#action2y1 = np.ones(1045) * 0.1
#action2rx = np.zeros(1000)

#ry
action2ry0 = np.zeros(5)
action2ry1 = np.ones(85) * v10
action2ry2 = np.ones(170) * -v10
action2ry3 = np.ones(85) * v10

action2ry0_ = np.zeros(10)
action2ry1_ = np.ones(70) * v10
action2ry2_ = np.ones(100) * -v10
action2ry3_ = np.ones(70) * v10

action2ry0__ = np.zeros(10)
action2ry1__ = np.ones(50) * v10
action2ry2__ = np.ones(100) * -v10
action2ry3__ = np.ones(50) * v10

action2ry4 = np.zeros(10)
action2ry5 = np.ones(50) * v10
action2ry6 = np.ones(100) * -v10
action2ry7 = np.ones(50) * v10

action2ry8 = np.zeros(10)
action2ry9 = np.ones(50) * v10
action2ry10 = np.ones(100) * -v10
action2ry11 = np.ones(50) * v10

action2ry12 = np.zeros(10)
action2ry13 = np.ones(50) * v10
action2ry14 = np.ones(100) * -v10
action2ry15 = np.ones(50) * v10

action2ry16 = np.zeros(25)
action2ry18 = np.ones(35) * -v10
action2ry19 = np.ones(70) * v10
action2ry20 = np.ones(35) * -v10

# action2ry21 = np.ones(100) * 0





action2rz0 = np.zeros(5)
action2rz1 = np.ones(85) * v20
action2rz2 = np.ones(170) * -v20*1.3
action2rz3 = np.ones(85) * v20

action2rz0_ = np.zeros(10)
action2rz1_ = np.ones(60) * v20
action2rz2_ = np.ones(120) * -v20*1.2
action2rz3_ = np.ones(60) * v20

action2rz0__ = np.zeros(10)
action2rz1__ = np.ones(50) * v20
action2rz2__ = np.ones(100) * -v20*1.2
action2rz3__ = np.ones(50) * v20



action2rz4 = np.zeros(10)
action2rz5 = np.ones(50) * v20
action2rz6 = np.ones(100) * -v20
action2rz7 = np.ones(50) * v20

action2rz8 = np.zeros(10)
action2rz9 = np.ones(50) * v22
action2rz10 = np.ones(100) * -v22
action2rz11 = np.ones(50) * v22

action2rz12 = np.zeros(10)
action2rz13 = np.ones(50) * -v23
action2rz14 = np.ones(100) * v23
action2rz15 = np.ones(50) * -v23

action2rz16 = np.zeros(25)
action2rz18 = np.zeros(35) * -v20
action2rz19 = np.zeros(70) * v20
action2rz20 = np.zeros(35) * -v20

# action2rz21 = np.zeros(100)


#action2rz=np.random.random(1000) * (np.random.random(1000)-0.5)*2

x = np.concatenate((action2x0, action2x1, action2x0_, action2x1_, action2x0__, action2x1__, action2x2,action2x3,
                    action2x4,action2x5,action2x6,action2x7,action2x8,action2x9,action2x10,action2x11)).reshape(-1, 1)
y = action2y1.reshape(-1, 1)
ry = np.concatenate(( action2ry0,action2ry1,action2ry2,action2ry3,
                     action2ry0_,action2ry1_,action2ry2_,action2ry3_,
                     action2ry0__,action2ry1__,action2ry2__,action2ry3__,
                      action2ry4,action2ry5,action2ry6,action2ry7,
                      action2ry8,action2ry9,action2ry10,action2ry11,
                      action2ry12,action2ry13,action2ry14,action2ry15,
                      action2ry16,action2ry18,action2ry19,action2ry20)).reshape(-1, 1)
rz = np.concatenate(( action2rz0,action2rz1,action2rz2,action2rz3,
                     action2rz0_,action2rz1_,action2rz2_,action2rz3_,
                     action2rz0__,action2rz1__,action2rz2__,action2rz3__,
                      action2rz4,action2rz5,action2rz6,action2rz7,
                      action2rz8,action2rz9,action2rz10,action2rz11,
                      action2rz12,action2rz13,action2rz14,action2rz15,
                      action2rz16,action2rz18,action2rz19,action2rz20)).reshape(-1, 1)
swing = np.concatenate((x, ry, rz), axis = 1)

ac_dir = '../../jaka/swing/left_and_right.npy'
# ob_dir = '/home/zhou/tactile_gym/tactile_gym/vae/train_data/obs.npy'
np.save(ac_dir,swing)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if model_test:
    obs = env.reset()
    sum_reward = 0
    f = h5py.File('/home/braynt/Tac/tactile_gym/tactile_gym/vae_trans/data_collect/_sim_pair_0.h5', 'w')        #simulation_all_pair1.h5
    for i in range(5):
        #f = h5py.File('/home/nathan/tactile_gym/tactile_gym/vae/train_data/episode_'+str(i)+'.h5', 'w')
        
        action_list = []
        obs_list = []
        obs = env.reset()
        #print(obs)
        print(len(swing))
        for i in range(len(swing)):
            #print(type(action))
            #action = env.action_space.sample()

            action = swing[i]
            #print(action)
            #print(action)
            f.create_dataset('data_'+str(i), data=env.current_img)
            # print("x:",obs["extended_feature"][0])
            # print("y:",obs["extended_feature"][1])
            # print("z:",obs["extended_feature"][2])
            # print("rx:",obs["extended_feature"][3])
            # print("ry:",obs["extended_feature"][4])
            # print("rz:",obs["extended_feature"][5])
            obs_list.append(obs["extended_feature"])
            obs, reward, done, info = env.step(action)
            # print(obs)
            sum_reward += reward

            if done:
                break
        # np.save(ob_dir,obs_list)
        break
