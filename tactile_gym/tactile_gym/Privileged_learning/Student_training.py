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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--student-model-dir', default='./model/push/', type=str, help='save student')
parser.add_argument('--expert-model-dir', default='../../model_checkpoints/push/best_model', type=str, help='load expert')

parser.add_argument('--load-student', default=True, type=bool)
parser.add_argument('--load-data', default=True, type=bool)

parser.add_argument('--test', default=False, type=bool)
parser.add_argument('--batch-size', default=256, type=int)
parser.add_argument('--collect-epochs', default=100, type=int)
parser.add_argument('--train-epochs', default=100, type=int)
parser.add_argument('--num-workers', default=10, type=int)
parser.add_argument('--learning-rate', default=0.0001, type=float)
parser.add_argument('--show-gui', default=False, type=bool)
parser.add_argument('--show-tactile', default=False, type=bool)

args = parser.parse_args()
seed = int(0)
max_steps = 1000
show_gui = args.show_gui
show_tactile = args.show_tactile
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

expert = TQC.load(args.expert_model_dir, env, device = device)


LR = args.learning_rate

model_dir = args.student_model_dir


criterion = torch.nn.MSELoss()
student_net = Student(use_gpu=True).to(device)
optimizer = optim.Adam(student_net.parameters(), lr=LR)

def train(data_all,student_net,train_epochs):
    student_net.train()
    train_dataset = dataset(data_set=data_all)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers)
    best_loss = 100
    for epoch in range(train_epochs):
        total_loss = 0.0
        for i, data in enumerate(train_data_loader):
            optimizer.zero_grad()  
            obs,action = data
            obs=obs.float().to(device)
            action=action.float().to(device)
            output = student_net(obs)
            loss = criterion(output,action)
            loss.backward() 
            optimizer.step()  
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
        
        #define noise pattern
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
            
            #mix policy
            prob = random.uniform(0,1)
            if prob > 0.5:
                action_mix = expert_action
            else: 
                action_mix = student_action

            data["label"] = expert_action
            data_set.append(data)
            
            obs, reward, done, info = env.step(action_mix)
            
            #add_noise
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
    print("====Testing======")
    avg_reward = 0
    for i in range(test_epoch):
        sum_reward = 0
        obs = env.reset()
        obs_data = np.hstack((obs["extended_feature"], obs["encodedimg"]))
        obs_squence = Queue(maxsize=8) #window_size = 8
        for j in range(8):
            obs_squence.put(obs_data)   
        for i in range(500):
            
            obs_batch = torch.tensor(np.array(obs_squence.queue)).unsqueeze(0).float().to(device)

            action = student_net(obs_batch).cpu().detach().numpy()[0]
            #print(action)

            obs, reward, done, info = env.step(action)
            #print(obs)
            obs_data = np.hstack((obs["extended_feature"], obs["encodedimg"]))
            obs_squence.get() 
            obs_squence.put(obs_data) 

            sum_reward += reward
            if done:
                break
        print("Test_reward: ",sum_reward)
        avg_reward += sum_reward 
        sum_reward = 0    
    avg_reward = avg_reward / 10
    print("Avg_reward: ", avg_reward)

    
if __name__ == "__main__":
    count = 0
    data_all = []
    
    print("================================")
    print("       Student Training")
    print("================================")
    # pretrain 
    if args.load_data:
        if os.path.exists(model_dir + 'data.pkl'):        
            with open(model_dir + 'data.pkl', 'rb') as file:
                data_all = pickle.load(file)
            print("Successfully loaded model")
        else:
            print("Data doesn't exist, collecting will begin")
            count,data_all = collect(env,student_net,data_all,collet_epoch=args.collect_epochs, count=0)
            with open('./model/push/data.pkl', 'wb') as file:
                pickle.dump(data_all, file)
    else:
        print("Collecting will begin")
        count,data_all = collect(env,student_net,data_all,collet_epoch=args.collect_epochs, count=0)
        with open('./model/push/data.pkl', 'wb') as file:
            pickle.dump(data_all, file)

    
    if args.load_student:
        if os.path.exists(model_dir + '_best'):
            student_net.load_state_dict(torch.load(model_dir+'_best'))
            print("Successfully loaded data")
        else:
            print("Model doesn't exist, training will begin")
            train(data_all,student_net,train_epochs=args.train_epochs)    
    else:
        print("Training will begin")
        train(data_all,student_net,train_epochs=args.train_epochs)
    

    #DAgger
    for i in range(10):
        print("iter: ",i)

        count, data_all = collect(env,student_net,data_all,collet_epoch=args.collect_epochs, count=count)
        with open('./model/push/data.pkl', 'wb') as file:
            pickle.dump(data_all, file)

        train(data_all,student_net,train_epochs=args.train_epochs)
        torch.save(student_net.state_dict(), model_dir+f"_{i}")
        if args.test:
            test(env,student_net,test_epoch=10)
        else:
            pass

            

