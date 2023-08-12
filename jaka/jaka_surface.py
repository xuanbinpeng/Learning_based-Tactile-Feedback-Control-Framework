# -*- coding: utf-8 -*-
#Author: Wu Qiwei 4/2/2023
#Contact: 1055482789@qq.com
import cv2
import cv2.aruco as aruco
import h5py
import threading
import torch
import time
import urx
import pybullet as pb
import sys
import gym
sys.path.append('/media/mo/MOPAN/handeye/jaka(1)/jaka/libx86_64-linux-gnu')
import torch
import time

#from libx86_64-linux-gnu import *
import jkrc
import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

from stable_baselines3.common.noise import NormalActionNoise
from sb3_contrib import TQC
# from tf import transformations
# from utils import *
from Privileged_learning.model import Student

from VAE_cross_domain.model import DisentangledVAE,Discriminator,FeatureMapping
from opensimplex import OpenSimplex
from queue import Queue
from action_transformation import *
import argparse
parser = argparse.ArgumentParser()


parser.add_argument('--collect_data',default=False,type=bool)
parser.add_argument('--show_tactile',default=False,type=bool)
parser.add_argument('--show_path',default=True,type=bool)

robot_get_goal = False

args = parser.parse_args()
# Constant definition
PI=3.1415926

ABS = 0
INCR = 1
CONT = 2

COORD_BASE = 0
COORD_JOINT = 1
COORD_TOOL = 2

Enable = True
Disable = False

T_calib = [-0.556598  ,  -0.131296  ,  0.58]
R_calib = [-0.984, -0.104, -0.067,0.131] #-164 6 -13

mtx = np.array([
        [608.4126046076422, 0.0, 318.77672353568124],
        [     0.0, 608.7173924297246, 243.77145734214503],
        [      0,       0,      1],
        ])
dist = np.array( [0.028784404196194664, 0.7481844381754564, 0.0028707604214314336, -0.0032153215725527914, -3.1796489988923713] )

T_base_cam = np.eye(4)
M_base_cam = np.array(pb.getMatrixFromQuaternion(R_calib)).reshape(3,3)
T_base_cam[:3, :3]=M_base_cam
T_base_cam[0][3] = T_calib[0]
T_base_cam[1][3] = T_calib[1]
T_base_cam[2][3] = T_calib[2]


Model_R = DisentangledVAE
Model_S = DisentangledVAE
Model_D = Discriminator
Model_M = FeatureMapping

# start device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
robot = jkrc.RC("10.5.5.100") #creat robot object
robot.login()
robot.power_on() 
robot.enable_robot()

# reset_position=[-0.18599,-0.45162,0.151,0.0121,-3.1593,-0.016] #   surface表面检测初始位置
reset_position = [-330, 0, 40, 0, 0, 0]

global real_action
global robot_goal

cap = cv2.VideoCapture(6)

tactile = cv2.VideoCapture(4)
tactile.set(cv2.CAP_PROP_FRAME_WIDTH,640)
tactile.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
tactile.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
tactile.set(cv2.CAP_PROP_FPS,100)

    
def check_limit(pos,action):
    limit = False
    next_x = pos[0]-0.25
    next_y = pos[1]+action[0]
    next_rz = pos[5]-action[1]*0.036
    if next_x < -660 or next_x > -320:
       limit = True       
    if next_y < -150 or next_y > 150:
       limit = True

    return limit


# main env

class JakaPushEnv():
    def __init__(self, draw_path = False, show_tactile = False, collect_data = False):
        self.draw_path = draw_path
        self.show_tactile = show_tactile
        self.init_pos = [-330, 0, 40, 0, 0, 0]
        self.collect_data = collect_data
        #x range : -360 ~ -560 -> -0.1 ~ 0.7
        #y range : 180 ~ -180 -> -0.4 ~ 0.4
        #rz range : -45 ~ 45 *pi / 180
    #-------------------------------------
        self.model_R = Model_R()
        self.model_S = Model_S()
        
        self.model_R.load_state_dict(torch.load("/media/mo/MOPAN/handeye/ur/ur5_push/VAE_cross_domain/checkpoints/0_model.pt",map_location=torch.device('cpu')))
        self.model_S.load_state_dict(torch.load("/media/mo/MOPAN/handeye/ur/ur5_push/VAE_cross_domain/checkpoints/1_model.pt",map_location=torch.device('cpu')))

        self.model_R = self.model_R.to(device)
        self.model_S = self.model_S.to(device)

    #-------------------------------------
        self.model_D = Model_D()
        self.model_M = Model_M()

        self.model_D.load_state_dict(torch.load("/media/mo/MOPAN/handeye/ur/ur5_push/VAE_cross_domain/checkpoints/discriminator_model.pt",map_location=torch.device('cpu')))
        self.model_M.load_state_dict(torch.load("/media/mo/MOPAN/handeye/ur/ur5_push/VAE_cross_domain/checkpoints/mapping_model.pt",map_location=torch.device('cpu')))
        #print(model_M)
        
        self.model_D = self.model_D.to(device)
        self.model_M = self.model_M.to(device)
        
        self.workframe_directions=[0,0,0]
        self.traj_n_points = 1    #     目标点？

        self.min_action, self.max_action = -0.25, 0.25
        self.termination_pos_dist = 0.01 #0.025

        # self.traj_n_points = 9
        self.traj_spacing = 0.03
        self.traj_max_perturb = 0.05

        self.termination_pos_dist = 0.07 #0.025
        self.obj_width = 0.075
        self.count = 0
        plt.ion()
        
    def get_tactile_img(self):
        ret,frame = tactile.read()
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame,(128,128))

        self.count=self.count+1
        return frame
    
    def get_encoded_img(self):
        frame_origin = self.get_tactile_img()
        frame = frame_origin / 255
        frame = frame.reshape((1,1,128,128))
        frame = torch.tensor(frame, dtype=torch.float32)
        frame = frame.to(device)
        
        mu, _ = self.model_R.encoder(frame)
        mu_sim = self.model_M(mu)
        #print(mu_sim)
        if self.show_tactile:
            fakeimage = self.model_S.decoder(mu_sim)
        
            fakeimage = fakeimage.cpu().detach().numpy()
            fakeimage = fakeimage.reshape((128, 128))
            fakeimage *= 255
            plt.subplot(121)
            plt.imshow(frame_origin)
            plt.subplot(122)
            plt.imshow(fakeimage)
            plt.pause(0.0001)
            plt.clf() 
        obs_vector = mu_sim.detach().cpu().numpy()[0]
        return obs_vector

    def get_extended_feature(self):
        obs = []
        
        ee_vec = robot.get_tcp_position()[1] #pos rpy 6
        trans = ee_vec[0]-reset_position[0]
        ee_vec[0] = -ee_vec[1]+reset_position[1]-0.008306622505187988
        ee_vec[1] = -trans-0.0009619250777177513
        ee_vec[2] = ee_vec[2]
        ee_vec[3] = 0
        ee_vec[4] = 0
        ee_vec[5] = ee_vec[5] +reset_position[5]-0.003047645092010498

        obs.extend(ee_vec)   
        # obs.extend(self.goal_next_pos)       
        # obs.extend(self.goal_next_rpy)
        

        return np.array(obs).astype(np.float32)

    def get_obs(self):
        observation = {}

        observation["encodedimg"] = self.get_encoded_img()
        observation["extended_feature"] = self.get_extended_feature()


        return observation
        
    def reset(self):
        robot.linear_move(self.init_pos,ABS,True,10)
        print("reset done!")

        obj_pos = np.zeros(3)

        self.traj_pos = np.zeros(3)
        self.traj_rpy = np.zeros(3)
        
        simplex_noise = OpenSimplex(seed=np.random.randint(1e8))
        #init_offset = self.obj_width / 2 + self.traj_spacing                             
        # generate smooth 1d traj using opensimple
        #生成目标点的坐标
        if(robot_get_goal):
            #self.traj_pos=robot_goal[0:3]
            trans = robot_goal[0]-reset_position[0]
            x=-robot_goal[1]+reset_position[1]-0.008306622505187988
            y=-trans-0.0009619250777177513
            z=robot_goal[2]
        else:
            x = 0
            y = 0.1
            z =reset_position[2]
        self.traj_pos = [x, y, z]      
        self.traj_rpy=[0,0,0]             #目标点位置
        
        print("goal_pos:",self.traj_pos)
        
        if (self.draw_path):            #画点
            plt.axis([-0.15,0.7,-0.5,0.5])
            #plt.plot(obj_pos[0] , obj_pos[1],marker='o',color='gold')
            plot_x = np.array(x)
            plot_y = np.array(y)
            #print(plot_x,plot_y)
            plt.plot(plot_x, -plot_y, marker='o',color='red')

        self.obj_path = []
        if self.collect_data:
            self.obj_path.append(cur_obj)


        #self.traj_rpy[:2] = np.gradient(self.traj_pos[:1],0.03)
        #self.traj_rpy=[reset_position[3],reset_position[4],reset_position[5]]
        self.make_goal()
        return self.get_obs()
    
    def step(self,action):

        ee_vec = robot.get_tcp_position()[1] #pos rpy 6
        #ee_vec[2] = 0
        trans = ee_vec[0]-reset_position[0]
        ee_vec[0] = -ee_vec[1]+reset_position[1]-0.008306622505187988
        ee_vec[1] = -trans-0.0009619250777177513
        ee_vec[2] = ee_vec[2]
        ee_vec[3] = 0
        ee_vec[4] = 0
        ee_vec[5] = ee_vec[5] +reset_position[5]-0.003047645092010498
        #print('%.9f' % ee_vec[0],'%.9f' % ee_vec[1],'%.9f' % ee_vec[5])
        #print(ee_vec[0],ee_vec[1],ee_vec[5])

        ee_pos = np.array([ee_vec[0],ee_vec[1],ee_vec[2]])

        self.obj_pos = np.zeros(3)
        #obj_pos = np.zeros_like(self.obj_pos)        
        self.obj_pos[0] = ee_vec[0]       #/ -1 -0.38
        self.obj_pos[1] = ee_vec[1]       #-0.06
        self.obj_pos[2] = ee_vec[2]

        obs = self.get_obs()
        reward = self.get_reward(self.obj_pos)
        done = self.is_terminated(ee_pos)
        info = {}

        if self.draw_path:


            plt.axis([-0.15,0.7,-0.4,0.4])           
            #plt.plot(obj_pos[0], obj_pos[1],marker='o',color='g')
            plt.plot(ee_vec[0], -ee_vec[1],marker='+',color='brown')
            plt.plot(self.goal_pos[0] , -self.goal_pos[1],marker='o',color='red')
            #plt.plot(self.goal_next_pos[0] , self.goal_next_pos[1],marker='o',color='blue')
            plt.pause(0.0001)
        

        if self.collect_data:
            self.obj_path.append(cur_obj)


        return obs,reward,done,info
        
    def get_reward(self,obj_pos):
        obj_goal_pos_dist = np.linalg.norm(obj_pos - self.goal_pos)
        #print(obj_goal_pos_dist)
        #obj_goal_orn_dist = np.arccos(np.clip((2 * (np.inner(self.goal_orn, self.obj_orn) ** 2))-1,-1,1))
        
        return 0
        
    def is_terminated(self,obj_pos):
        obj_goal_pos_dist = np.linalg.norm(obj_pos - self.goal_pos)
        #print("obj_pos:",obj_pos,"self.goal_pos:",self.goal_pos)
        #print("obj_goal_pos_dis:",obj_goal_pos_dist)
        if obj_goal_pos_dist < self.termination_pos_dist:
             return True
        #if max_step
        return False
    def make_goal(self):
            self.goal_pos = self.traj_pos
            self.goal_rpy = self.traj_rpy
    def encode_actions(self, actions):
        """
        return actions as np.array in correct places for sending to ur5
        """
        encoded_actions = np.zeros(6)
        self.workframe_directions[1]=np.random.choice([-1,1])
        encoded_actions[1] = self.workframe_directions[1] * 0.25
        encoded_actions[0] = actions[0]
        encoded_actions[5] = actions[1]

        return encoded_actions



def control_thread():
    # print('con')
    while True:
        if real_action is not None:
            
            # print(real_action)
            robot.servo_p(cartesian_pose = real_action,move_mode = INCR)
            # robot.speedl([-real_action[1]*0.94,-real_action[0]*1.057,0,0,0,-real_action[5]*1.294],0.5,0.2)
            pass
            #robot.servo_p(cartesian_pose = real_action,move_mode = INCR)
        time.sleep(0.008)      

def camera_thread():
    global cur_obj
    cur_obj = None
    #cur_obj=[]
    
    while True:
        ret, frame = cap.read()   
        # print('111111')
        cv2.imshow('frame',frame)
        print(frame.shape)
        cv2.waitKey(1)
        #cur_obj = [0,0,0,0,0,0]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #print('end1')
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        #print('end2')
        parameters =  aruco.DetectorParameters_create()
        #print('end3')
        cv2.imshow('gray', gray)
        cv2.waitKey(1)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray,
                                                          aruco_dict,
                                                          parameters=parameters)
        
        if ids is not None:
            for i in range(len(ids)):
                if ids[i][0] == 1:
                    rvec_obj, tvec_obj, _ = aruco.estimatePoseSingleMarkers(corners[i], 0.04, mtx, dist)

                    (rvec_obj-tvec_obj).any()
                    M_cam_obj, _ = cv2.Rodrigues(rvec_obj)
                    T_cam_obj = np.eye(4)
                    T_cam_obj[:3, :3]=M_cam_obj
                    T_cam_obj[0][3] = tvec_obj[0][0][0]
                    T_cam_obj[1][3] = tvec_obj[0][0][1]
                    T_cam_obj[2][3] = tvec_obj[0][0][2]
                    
                    T_base_obj = np.dot(T_base_cam,T_cam_obj)
                    
                    obj_orn_q = Quaternion(matrix=T_base_obj[:3, :3])

                    obj_orn = np.array([obj_orn_q.x,obj_orn_q.y,obj_orn_q.z,obj_orn_q.w])
                    
                    obj_pos = np.array([T_base_obj[0][3],T_base_obj[1][3],T_base_obj[2][3]])

                    cur_obj = [obj_pos,obj_orn]
                    print(cur_obj)
                    # print(tvec_obj)
                else:
                    continue
        else:
            continue 

#main
if __name__ == '__main__':
    if robot_get_goal:
        print("控制机械臂移动到目标点,按y确定目标点,按n退出采点:")
        while True:
            input_str = input()
            if (input_str=='y'):
                robot_goal = robot.getl()
                #print(robot_goal)
                break
            elif (input_str == 'n'):
                robot_get_goal=False
                break
            else:
                pass
    model_dir = '/media/mo/MOPAN/handeye/ur/ur5_push/Privileged_learning/model/push/'
    env = JakaPushEnv(draw_path = args.show_path,show_tactile=args.show_tactile,collect_data=args.collect_data)
    model = Student(input_size=38,use_gpu=False)
    # model.load_state_dict(torch.load(model_dir+"_14",map_location=torch.device('cpu')))

    thread_cam = threading.Thread(target=camera_thread, daemon=True)
    thread_cam.start()

    while True:
        #print(cur_obj)
        if cur_obj is None:
            time.sleep(0.001)
        else:
            break

    obs = env.reset()

    obs_data = np.hstack((obs["extended_feature"], obs["encodedimg"]))
    obs_squence = Queue(maxsize=8) #队列
    for j in range(8):
        obs_squence.put(obs_data) 

    robot.servo_move_enable(Enable)
    time.sleep(0.5)

    is_limit = False

    plt.ion()

    real_action = None
    thread = threading.Thread(target=control_thread, daemon=True)
    thread.start()
    time_avg = 0
    # ac_dir = '/home/zhou/tactile_gym/tactile_gym/Privileged_learning/action/surface/surface_action.npy'
    # swing=np.load(ac_dir)
    for i in range(200):
        time1 = time.time()
        #print(obs_squence.queue)
        obs_batch = torch.tensor(obs_squence.queue).unsqueeze(0).float()
        #print(obs_batch)
        action = model(obs_batch).detach().numpy()[0]
        ret,pos = robot.get_tcp_position()
        cur_tcp_orn = pb.getQuaternionFromEuler([pos[3],pos[4],pos[5]])
        #print(action)
        #encoded_action = encode_TCP_frame_actions(action,cur_tcp_orn)
        encoded_action = env.encode_actions(action)
        #real_action = scale_actions(encoded_action) 
        # real_action=swing[i]
        real_action = scale_actions(encoded_action) 
        real_action = [real_action[0]*-11.5,real_action[1]*11.5,real_action[2],real_action[3],real_action[4],real_action[5]*0.0116]
        print(real_action)
        #print(real_action)    
        #is_limit = check_limit(pos,action)       
        #if is_limit:
        #    break
        obs, reward, done, info = env.step(action)
        #print(i,":",obs["extended_feature"][0],obs["extended_feature"][1],obs["extended_feature"][5])
        #GUANJIAN print(obs["extended_feature"][0])
        obs_data = np.hstack((obs["extended_feature"], obs["encodedimg"]))
        obs_squence.get() #删除第一张图
        obs_squence.put(obs_data) #插入现在的图

        if done:
            break
        time2 = time.time()
        time_wait = 0.1 - (time2-time1)
        if time_wait > 0:
            time.sleep(time_wait)
        else:
            pass
    if env.collect_data == True:
        print(env.obj_path)
        #np.savetxt('/home/zhou/tactile_gym/examples/test_path_pos.npy',env.obj_path[:][0])
        #np.savetxt('/home/zhou/tactile_gym/examples/test_path_orn.npy',env.obj_path[:][1])
        pass
    else:
        pass
    time.sleep(0.008)
    robot.stopl(20)
    plt.ioff()
    #plt.show()

    real_action = None
    print("ok")
        
        
        
    
