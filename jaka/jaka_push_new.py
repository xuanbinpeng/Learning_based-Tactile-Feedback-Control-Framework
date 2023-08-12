# -*- coding: utf-8 -*-
#Author: Wu Qiwei 4/2/2023
#Contact: 1055482789@qq.com
import os
import cv2
import cv2.aruco as aruco
import h5py
import threading
import torch
import time
import pybullet as pb
import sys
import gym
sys.path.append('/home/braynt/Tac/jaka/libx86_64-linux-gnu')
import torch
import time
#export JAKA_LIBRARY_PATH=/home/braynt/Tac/jaka/libx86_64-linux-gnu
#export LD_LIBRARY_PATH=/home/braynt/Tac/jaka/libx86_64-linux-gnu:$LD_LIBRARY_PATH 

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

#TODO test the argparse   main
parser.add_argument('--collect_data',default=False,type=bool)
parser.add_argument('--show_tactile',default=False,type=bool)
parser.add_argument('--show_path',default=False,type=bool)

video_path = 'test.avi'
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

#TODO check the calib matrix
R_calib = [ -0.99980572, -0.01755857, -0.00864419, 0.00234615 ]
T_calib = [-6.838917942945998973e-01, -2.664004708178525221e-02, 6.070689492756984329e-01] #-164 6 -13

mtx = np.array([[909.65582275, 0, 652.90588379],
[ 0, 907.89691162, 365.95373535],
[ 0, 0, 1. ]])
# dist = np.array( [0.028784404196194664, 0.7481844381754564, 0.0028707604214314336, -0.0032153215725527914, -3.1796489988923713] )
dist = np.array( [0.000000, 0.000000, 0.000000, 0.000000, 0.000000] )

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

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

robot = jkrc.RC("10.5.5.100") #creat robot object
robot.login()
robot.power_on() 
robot.enable_robot()
# robot.set_tcp((0,-0.14,0.04,0,0,0))
# robot.set_payload(2,(0,0,0.1))

reset_position = [-330, 0, 40, 0, 0, 0]
global real_action

#TODO before testing check the camera number
cap = cv2.VideoCapture(8)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FPS,60)
video_writer = cv2.VideoWriter(video_path,cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),cap.get(cv2.CAP_PROP_FPS),(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

tactile = cv2.VideoCapture(2)
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
    #if next_rz < -45/180*PI or next_rz > 45/180*PI:
    #   limit = True   
    return limit

# main env
class JakaPushEnv():
    def __init__(self, draw_path = False, show_tactile = False, collect_data = False):
        self.draw_path = draw_path
        self.show_tactile = show_tactile
        self.init_pos = [-330, 0, 30, 0, 0, 0]
        self.collect_data = collect_data
        #x range : -360 ~ -560 -> -0.1 ~ 0.7
        #y range : 180 ~ -180 -> -0.4 ~ 0.4
        #rz range : -45 ~ 45 *pi / 180
    #-------------------------------------
        self.model_R = Model_R()
        self.model_S = Model_S()
        
        self.model_R.load_state_dict(torch.load("/home/braynt/Tac/jaka/VAE_cross_domain/checkpoints/0_model.pt"))
        self.model_S.load_state_dict(torch.load("/home/braynt/Tac/jaka/VAE_cross_domain/checkpoints/1_model.pt"))

        self.model_R = self.model_R.to(device)
        self.model_S = self.model_S.to(device)

    #-------------------------------------
        self.model_D = Model_D()
        self.model_M = Model_M()

        self.model_D.load_state_dict(torch.load("/home/braynt/Tac/jaka/VAE_cross_domain/checkpoints/discriminator_model.pt"))
        self.model_M.load_state_dict(torch.load("/home/braynt/Tac/jaka/VAE_cross_domain/checkpoints/mapping_model.pt"))
        
        self.model_D = self.model_D.to(device)
        self.model_M = self.model_M.to(device)
        
        self.traj_n_points = 10
        self.traj_spacing = 0.03
        self.traj_max_perturb = 0.15

        self.traj_type="s1in"    #new     alter:  straight sin random ....
        self.loadpath=3          #new

        self.obj_pos = np.zeros(3)  #new
        self.termination_pos_dist = 0.01 #0.025
        self.reset_rpy=[]
        self.obj_width = 0.075
        self.count = 0
        plt.ion()
        
    def get_tactile_img(self):
        ret,frame = tactile.read()
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        frame = frame[80:480,100:530]

        (h, w) = frame.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), 132, 1.0) 
        frame = cv2.warpAffine(frame, M, (w, h))
        frame = cv2.resize(frame,(128,128))
        rett,frame = cv2.threshold(frame,60,255,cv2.THRESH_BINARY)

        self.count=self.count+1
        return frame
    
    def get_encoded_img(self):

        frame_origin = self.get_tactile_img()
        frame = frame_origin / 255
        frame = frame.reshape((1,1,128,128))
        frame = torch.tensor(frame, dtype=torch.float32).to(device)
        
        mu, _ = self.model_R.encoder(frame)
        mu_sim = self.model_M(mu)

        if self.show_tactile:  #TODO check the transformed fakeimg
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


        #Tcp_position   should be correspond to sim environment 
        ee_vec = robot.get_tcp_position()[1]
        ee_vec[0] = (-ee_vec[0]+reset_position[0])*0.0009
        ee_vec[1] = ee_vec[1]*0.00063   
        ee_vec[2] = 0
        ee_vec[3] = 0
        ee_vec[4] = 0
        ee_vec[5] = -ee_vec[5]

        print('tcp',ee_vec[0],ee_vec[1],ee_vec[5])
        #TODO print(ee_vec[0],ee_vec[1],ee_vec[5])

        obs.extend(ee_vec) 
        obs.extend(self.goal_pos)
        obs.extend(self.goal_rpy)     
        obs.extend(self.goal_next_pos)       
        obs.extend(self.goal_next_rpy)
        

        return np.array(obs).astype(np.float32)

    def get_obs(self):
        observation = {}
        
        observation["encodedimg"] = self.get_encoded_img()

        observation["extended_feature"] = self.get_extended_feature()

        return observation
        
    def reset(self):
        robot.linear_move(self.init_pos,ABS,True,50)
        print("reset done!")

        #obj_pos = cur_obj[0]
        obj_pos = np.zeros(3)

        self.traj_pos = np.zeros(shape=(self.traj_n_points,3))
        self.traj_rpy = np.zeros(shape=(self.traj_n_points,3))
        
        self.traj_id = -1
        
        simplex_noise = OpenSimplex(seed=np.random.randint(1e8))
        init_offset = self.obj_width / 2 + self.traj_spacing

        # generate smooth 1d traj using opensimplex
        # new
        first_run = True
        rx = []
        ry = []

        if self.traj_type == "random":
            for i in range(int(self.traj_n_points)):

                noise = simplex_noise.noise2(x=i * 0.1, y=1) * self.traj_max_perturb

                if first_run:
                    init_noise_pos_offset = -noise
                    first_run = False

                x = init_offset + (i * self.traj_spacing)
                rx.append(x)
                y = init_noise_pos_offset + noise
                ry.append(y)
                z = 0.0
                self.traj_pos[i] = [x, y, z]

            if (self.draw_path):
                plt.axis([-0.15,0.7,-0.4,0.4])
                plt.plot(obj_pos[0] , obj_pos[1],marker='o',color='gold')
                plot_x = np.array(rx)
                plot_y = np.array(ry)
                plt.plot(plot_x, plot_y, '*r')
            self.traj_rpy[:,2] = np.gradient(self.traj_pos[:,1],0.03)


        elif self.traj_type == "sin":
            flip = True
            x = np.arange(init_offset, init_offset + (self.traj_n_points * self.traj_spacing), self.traj_spacing)
            y = np.sin((x - init_offset) / 0.2 * PI) * 0.04
            if flip:
                y = -y
            z = np.zeros_like(x)
            for i in range(int(self.traj_n_points)):
                self.traj_pos[i] = [x[i],y[i],z[i]]
            if (self.draw_path):
                plt.axis([-0.1,0.3,-0.4,0.4])
                # print(obj_pos)
                plt.plot(obj_pos[0] , obj_pos[1],marker='o',color='gold')
                plot_x = np.array(x)
                plot_y = np.array(y)
                plt.plot(plot_x, plot_y, '*r')
            self.traj_rpy[:,2] = np.gradient(self.traj_pos[:,1],0.03)

        elif self.traj_type == "straight":
            for i in range(int(self.traj_n_points)):

                noise = 0.008
                if first_run:
                    y = 0
                    first_run = False

                x = init_offset + (i * self.traj_spacing)
                rx.append(x)
                if not(first_run):
                    y += noise

                ry.append(y)
                z = 0.0
                self.traj_pos[i] = [x, y, z]

            if (self.draw_path):
                plt.axis([-0.15,0.7,-0.4,0.4])
                plt.plot(obj_pos[0] , obj_pos[1],marker='o',color='gold')
                plot_x = np.array(rx)
                plot_y = np.array(ry)
                plt.plot(plot_x, plot_y, '*r')
            self.traj_rpy[:,2] = np.gradient(self.traj_pos[:,1],0.03)
 

        elif self.traj_type== "load":
            load_air='/home/bryant/Tac/tactile_gym/tactile_gym/Privileged_learning/path/push/student/'+str(self.loadpath)+'_path.txt'
            path=np.loadtxt(load_air)
            #print(path[0])
            for i in range(len(path)):
                self.traj_pos[i][0:3] = path[i][0:3]
                self.traj_pos[i][1]=-self.traj_pos[i][1]
                self.traj_rpy[i][0:3] = path[i][3:6]
                rx.append(self.traj_pos[i][0])
                ry.append(self.traj_pos[i][1])
            if (self.draw_path):
                plt.axis([-0.15,0.7,-0.4,0.4])
                plt.plot(obj_pos[0] , obj_pos[1],marker='o',color='gold')
                plot_x = np.array(rx)
                plot_y = np.array(ry)
                plt.plot(plot_x,plot_y, '*r')


        else:
            x = np.arange(init_offset, init_offset + (self.traj_n_points * self.traj_spacing), self.traj_spacing)
            y = np.zeros_like(x)
            z = np.zeros_like(x)
            for i in range(int(self.traj_n_points)):
                self.traj_pos[i] = [x[i],y[i],z[i]]
            if (self.draw_path):
                plt.axis([-0.15,0.7,-0.4,0.4])
                plt.plot(obj_pos[0] , obj_pos[1],marker='o',color='gold')
                plot_x = np.array(x)
                plot_y = np.array(y)
                plt.plot(plot_x, plot_y, '*r')
            self.traj_rpy[:,2] = np.gradient(self.traj_pos[:,1],0.03)

        self.update_goal()
        return self.get_obs()
    
    def step(self,action):

        if(cur_obj != None):
            self.obj_pos = cur_obj[0:6]
            #TODO print(self.obj_pos)
            # self.obj_pos[0]=-(self.obj_pos[0]-0.169-0.05)
            # self.obj_pos[1]=-(self.obj_pos[1]+0.00612)
            # self.obj_pos[2] = 0

        # print(ee_vec[0],ee_vec[1],ee_vec[5])
            # print("obj",self.obj_pos[0:3])
            self.obj_pos[0]=-(self.obj_pos[0]-0.133-0.075/2)
            self.obj_pos[1]=-(self.obj_pos[1])
            self.obj_pos[2] = 0
            print("1obj",self.obj_pos[0:3])

        obs = self.get_obs()

        reward = self.get_reward(self.obj_pos)
        ee_pos = obs["extended_feature"][0:3]
        done = self.is_terminated(self.obj_pos)
        info = {}     

        if self.draw_path:

            # print(self.goal_pos[0])
            plt.axis([-0.1,0.3,-0.4,0.4])           
            plt.plot(ee_pos[0], -ee_pos[1],marker='+',color='brown')
            plt.plot(self.goal_pos[0] , self.goal_pos[1],marker='o',color='blue')
            # plt.plot(self.goal_next_pos[0] , self.goal_next_pos[1],marker='o',color='blue')
            plt.pause(0.0001)

        #new if self.collect_data:
            # print(cur_obj)
        #    self.obj_path.append(cur_obj)
        return obs,reward,done,info
        
    def get_reward(self,obj_pos):
        obj_goal_pos_dist = np.linalg.norm(obj_pos[0:3] - self.goal_pos)
        
        return 0
        
    def is_terminated(self,obj_pos):
        #obj_goal_pos_dist = np.linalg.norm(obj_pos - self.goal_pos)
        obj_goal_pos_dist = np.abs(obj_pos[0] - self.goal_pos[0])
        if obj_goal_pos_dist < self.termination_pos_dist:
            goal_updated = self.update_goal()
            print(self.goal_pos[0],obj_pos[0])
            if not goal_updated:
                return True
        return False
    
    def update_goal(self):
        self.traj_id += 1
        if self.traj_id >= self.traj_n_points:
            return False
        else:
            self.goal_pos = self.traj_pos[self.traj_id]
            self.goal_rpy = self.traj_rpy[self.traj_id]
        
            if self.traj_id + 1 < self.traj_n_points:
                self.goal_next_pos = self.traj_pos[self.traj_id+1]
                self.goal_next_rpy = self.traj_rpy[self.traj_id+1]
           
            else:
                self.goal_next_pos = self.traj_pos[self.traj_id]
                self.goal_next_rpy = self.traj_rpy[self.traj_id]
            return True       

def control_thread():
    while True:
        if real_action is not None:
            robot.servo_p(cartesian_pose = real_action,move_mode = INCR)
            pass
        time.sleep(0.008)    

def camera_thread():
    global cur_obj
    cur_obj = None
    
    while True:
        ret, frame = cap.read()   
        video_writer.write(frame)

        # cv2.imshow('frame',frame)
        # cv2.waitKey(1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        parameters =  aruco.DetectorParameters()
        # cv2.imshow('gray', gray)
        # cv2.waitKey(1)
        detector = aruco.ArucoDetector(aruco_dict,parameters)
        corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
        #print(ids)

        #TODO check id print(ids)

        if ids is not None:
            for i in range(len(ids)):
                if ids[i][0] == 2:
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

                    cur_obj = [obj_pos[0]+0.5,obj_pos[1],obj_pos[2],obj_orn[0],obj_orn[1],obj_orn[2],obj_orn[3]]
                    # print(cur_obj) #TODO check the cur_obj position and rpy

                else:
                    continue
        else:
            continue 

##################################main
if __name__ == '__main__':
    model_dir = '/home/braynt/Tac/jaka/Privileged_learning/model/'
    env = JakaPushEnv(draw_path = args.show_path,show_tactile=args.show_tactile,collect_data=args.collect_data)
    model = Student(use_gpu=True)
    model.load_state_dict(torch.load(model_dir+"_best"))

    model = model.to(device)
    thread_cam = threading.Thread(target=camera_thread, daemon=True)
    thread_cam.start()

    while True:
        if cur_obj is None:
            time.sleep(0.002)
        else:
            break
    
    obs = env.reset()  
    obs_data = np.hstack((obs["extended_feature"], obs["encodedimg"]))
    obs_squence = Queue(maxsize=8) #队列
    for j in range(8):
        obs_squence.put(obs_data) 
    #print(env.obs_time)
    robot.servo_move_enable(Enable)
    time.sleep(0.5)
    
    is_limit = False

    plt.ion()
    real_action = None

    thread = threading.Thread(target=control_thread, daemon=True)
    thread.start()
    time_avg = 0
    #new
    obj_path = []
    last_cur_obj=[0,0,0,0,0,0]
    obj_path.append(last_cur_obj)
    path_record=[]
    m=[]
    for i in range(10):
        m[0:3]=np.array(env.traj_pos[i][0:3])
        m[3:6]=np.array(env.traj_rpy[i][0:3])
        #print(m)
        path_record.append(np.array(m))  


    


    for i in range(240): # 200        
        time1 = time.time()
        obs_batch = torch.tensor(np.array(obs_squence.queue)).unsqueeze(0).float().to(device)
        action = model(obs_batch).detach().cpu().numpy()[0]
        print(action)
        # action = [0.1,0]
        ret,pos = robot.get_tcp_position()
        cur_tcp_orn = pb.getQuaternionFromEuler([pos[3],pos[4],pos[5]])
        encoded_action = encode_TCP_frame_actions(action,cur_tcp_orn)
        real_action = scale_actions(encoded_action) 

        #TODO check the direction and movement action = [0.1,0.1]
        real_action = [real_action[0]*-11.5,real_action[1]*11.5,real_action[2],real_action[3],real_action[4],real_action[5]*-0.0116] 

        # tcp : print(obs["extended_feature"][0:6])  goal [6:12] next goal[12:18]
        # obs[encodeimg][0:32] 

        #obs
        obs, reward, done, info = env.step(action)

        if (i == 0): 
            plt.plot(env.obj_pos[0], env.obj_pos[1],marker='s',markersize=20,alpha=0.5,color='b')
        #obj pos 

        if env.draw_path:
            if(np.linalg.norm(np.array(last_cur_obj[0:3]) - np.array(env.obj_pos[0:3]))>0.015):
                plt.plot(env.obj_pos[0], env.obj_pos[1],marker='s',markersize=20,alpha=0.5,color='b')
                last_cur_obj[0:6]=env.obj_pos
                obj_path.append(np.array(env.obj_pos))


        obs_data = np.hstack((obs["extended_feature"], obs["encodedimg"]))
        obs_squence.get() 
        obs_squence.put(obs_data)

        if done:
            break
        time2 = time.time()
        time_wait = 0.1 - (time2-time1)

        if time_wait > 0:
            time.sleep(time_wait)
        else:
            pass

    real_action = None

    #TODO  complete the data collecting program  ur5
    if env.collect_data == True:
    
        path_dir = './path_data/'
        obj_path_file = path_dir + 'obj_path' + '.txt'              #+ str(env.loadpath)
        path_record_file = path_dir + 'path' + '.txt'
     
        # if os.path.exists(obj_path_file):
        #     np.savetxt(obj_path_file, obj_path)
        # else:
        #     with open(obj_path_file, 'ab') as f:
        #         np.savetxt(f, obj_path)     
        #         print("saved1")      

        # if os.path.exists(path_record_file):
        #     np.savetxt(path_record_file, path_record)
        # else:
        #     with open(obj_path_file, 'ab') as f:
        #         np.savetxt(f, path_record)  
        #         print("saved2")      
        
        np.savetxt(obj_path_file, obj_path)
        np.savetxt(path_record_file, path_record)

        print("save done")

        # print(env.obj_path)
        # ac_dir = '/media/mo/MOPAN/handeye/ur/ur5_push/Privileged_learning/obj_path.npy'
        # np.save(ac_dir,env.obj_path)  


        #np.savetxt('/home/zhou/tactile_gym/examples/test_path_pos.npy',env.obj_path[:][0])
        #np.savetxt('/home/zhou/tactile_gym/examples/test_path_orn.npy',env.obj_path[:][1])
        pass
    else:
        pass
    
    time.sleep(0.008)
    robot.servo_move_enable(Disable)
    plt.ioff()
    env.reset()

    robot.motion_abort()
    print("ok")
        
        
        
    
