from tactile_gym.rl_envs.nonprehensile_manipulation.object_push.object_push_env import (
    ObjectPushEnv,
)
import torch
import pickle
import os
from torch import optim
from torch.utils.tensorboard import SummaryWriter 
from torch.utils.data.dataloader import DataLoader

import numpy as np
from tqdm import tqdm
import random
from queue import Queue
from data_loader import dataset
from model import Student
from sb3_contrib import TQC

import time

seed = int(0)
max_steps = 1000
show_gui = False
model_test = True
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
    #    "traj_type": "straight",
    'traj_type': 'simplex',

    # which observation type to return
    # 'observation_mode':'oracle',
    #"observation_mode": "tactile_and_feature",
    # 'observation_mode':'visual_and_feature',
    # 'observation_mode':'visuotactile_and_feature',
    # 'observation_mode':'encodedimg_and_feature',
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
obs_dim_dict = env.get_obs_dim()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

expert = TQC.load("../../model_checkpoints/push/best_model", env, device = device)



LR = 0.0001

model_dir = './model/push/'

#定义交叉熵损失函数
criterion = torch.nn.MSELoss()
student_net = Student(input_size = obs_dim_dict["encodedimg"][0] + obs_dim_dict["extended_feature"][0], use_gpu=True).to(device)
optimizer = optim.Adam(student_net.parameters(), lr=LR)

def train(data_all,student_net,train_epochs):
    student_net.train()
    train_dataset = dataset(data_set=data_all)
    train_data_loader = DataLoader(train_dataset, batch_size=256, shuffle=True,num_workers=10)
    best_loss = 100
    for epoch in range(train_epochs):
        total_loss = 0.0
        for i, data in enumerate(train_data_loader):
            optimizer.zero_grad()  # 清空梯度缓存
            obs,action = data
            obs=obs.float().to(device)
            action=action.float().to(device)
            output = student_net(obs)
            loss = criterion(output,action)
            loss.backward() # 反向传播
            optimizer.step()  # 更新参数
            total_loss += loss.item()
        train_loss = total_loss / len(train_data_loader.dataset)
        if (epoch % 10 == 0):
            print(f"eppch{epoch}, train_loss{train_loss:.8f}")
        #writer.add_scalar('train_loss', train_loss, global_step=epoch , walltime=None)
            if best_loss > train_loss:
                best_loss = train_loss
                torch.save(student_net.state_dict(), model_dir+"_best")


def collect(env,student_net, data_set,collet_epoch,count):
    student_net.eval()
    obs = env.reset()

    noise_dim = random.randint(0,32)
    noise_list = [0]*noise_dim + [1]*(32 - noise_dim)
    random.shuffle(noise_list)
    noise_value = (np.random.rand(32) * 2 - 1 ) * 0.02
    noise = noise_list * noise_value

    obs_data = np.hstack((obs["extended_feature"], obs["encodedimg"]+noise))
    sum_reward = 0
    for i in tqdm(range(collet_epoch)):

        noise_dim = random.randint(0,32)
        noise_list = [0]*noise_dim + [1]*(32 - noise_dim)
        random.shuffle(noise_list)

        count += 1
        data = {}
        obs = env.reset()
        data["train_data"] = None
        data["label"] = None

        obs_squence = Queue(maxsize=8) 
        
        for j in range(8):
            obs_squence.put(obs_data)   
        for i in range(500):
            
            data["train_data"] = np.array(obs_squence.queue)
            obs_batch = torch.tensor(data["train_data"]).unsqueeze(0).float().to(device)

            with torch.no_grad():
                student_action = student_net(obs_batch).cpu().numpy()[0]
                expert_action,_ = expert.predict(obs)
            
            prob = random.uniform(0,1)
            if prob > 0.5:
                action_mix = expert_action
            else: 
                action_mix = student_action
            #action_mix = expert_action * 0.5 + student_action * 0.5

            data["label"] = expert_action
            data_set.append(data)
            
            obs, reward, done, info = env.step(action_mix)

            noise_value = (np.random.rand(32) * 2 - 1 ) * 0.02
            noise = noise_list * noise_value
            obs_data = np.hstack((obs["extended_feature"], obs["encodedimg"]+noise))
            obs_squence.get() 
            obs_squence.put(obs_data) 

            sum_reward += reward  

            if done:
                break
        sum_reward = 0    

    return count, data_set

def test(env,student_net,test_epoch):
    avg_reward = 0
    for i in range(test_epoch):
        sum_reward = 0
        obs = env.reset()
        obs_data = np.hstack((obs["extended_feature"], obs["encodedimg"]))
        obs_squence = Queue(maxsize=8) #队列
        for j in range(8):
            obs_squence.put(obs_data)   
        for i in range(500):
            
            obs_batch = torch.tensor(np.array(obs_squence.queue)).unsqueeze(0).float().to(device)

            action = student_net(obs_batch).cpu().detach().numpy()[0]
            #print(action)

            obs, reward, done, info = env.step(action)
            #print(obs)
            obs_data = np.hstack((obs["extended_feature"], obs["encodedimg"]))
            obs_squence.get() #删除第一张图
            obs_squence.put(obs_data) #插入现在的图

            sum_reward += reward
            if done:
                break
        print(sum_reward)
        avg_reward += sum_reward 
        sum_reward = 0    
    avg_reward = avg_reward / 10
    print("Avg_reward: ", avg_reward)

    
if __name__ == "__main__":
    count = 0
    data_all = []

    count,data_all = collect(env,student_net,data_all,collet_epoch=200,count=0)
    train(data_all,student_net,train_epochs=100)

    for i in range(10):
        print("iter: ",i)

        count, data_all = collect(env,student_net,data_all,collet_epoch=100,count=count)
        train(data_all,student_net,train_epochs=100)

        #test(env,student_net,test_epoch=10)
        torch.save(student_net.state_dict(), model_dir+f"_{i}")

            

