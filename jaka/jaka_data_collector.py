import cv2
import h5py
import sys
sys.path.append('/home/braynt/Tac/jaka/libx86_64-linux-gnu')
import jkrc
import numpy as np
import threading
import time
import os
import random

from tqdm import tqdm
from action_transformation import *
from utils import *

global real_action

PI=3.1415926

Enable = True
Disable = False

ABS = 0
INCR = 1
CONT = 2

init_pos = [-420, 0, 140, 0, 0, 0]

robot = jkrc.RC("10.5.5.100") #creat robot object
robot.login()
robot.power_on() 
robot.enable_robot()

tactile = cv2.VideoCapture(2)
tactile.set(cv2.CAP_PROP_FRAME_WIDTH,640)
tactile.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
tactile.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
tactile.set(cv2.CAP_PROP_FPS,100)

f = h5py.File('/home/braynt/Tac/jaka/data_new/_real_pair_0_2.h5', 'w')
batch_size = 32
img_size = 128
count = 0

def control_thread():
    while True:
        if real_action is not None:
            pass
            robot.servo_p(cartesian_pose = real_action,move_mode = INCR)
        time.sleep(0.008) 

def load_data(data_file_path,batch_size,img_size):
    print("START LOADING H5")
    h5f_data = h5py.File(os.path.join(os.path.dirname(__file__), data_file_path), "r")
    print(len(h5f_data))
    batch_num = int(len(h5f_data) / batch_size)

    print("len(h5f_data) = ",len(h5f_data))
    np_data = np.zeros((len(h5f_data), img_size, img_size), dtype=np.uint8)

    for i in tqdm(range(len(h5f_data))):
        dset = h5f_data["data_"+str(i)]
        np_data[i] = np.asarray(dset[:])
        
    for i in range(len(np_data)):
        cv2.imshow("data",np_data[i])
        print('np_data[i]')
        time.sleep(0.05)
        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyWindow("data")
                   

if __name__ == "__main__":
    action_list = np.load("/home/braynt/Tac/jaka/swing/rolate_swing_all_pair.npy")  #employ the actions or design your own
    print(robot.get_tcp_position())
    robot.linear_move(init_pos,ABS,True,50)
    print(0)
    real_action = None
    robot.servo_move_enable(Enable)
    thread = threading.Thread(target=control_thread, daemon=True)
    thread.start()
    direction = random.choice((-1,1))
    random_data = True
    for i in range(1600):
        time1 = time.time()
        ret,frame = tactile.read()
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        
        frame = frame[80:480,100:530]
        (h, w) = frame.shape[:2]
        (cX, cY) = (w//2, h//2)
        M = cv2.getRotationMatrix2D((cX, cY), 132, 1.0)
        frame = cv2.warpAffine(frame, M, (w, h))
        frame = cv2.resize(frame,(128,128))
        rett, frame  =cv2.threshold(frame, 60, 255, cv2.THRESH_BINARY)
        cv2.imshow('Tactile',frame)
        f.create_dataset('data_'+str(count), data=frame) 
        count += 1
        action = action_list[i]
        # action=[0.1,0,0]
        print(action)
        # if random_data:s
        #     action[0] = action[0] * (0.75+random.random()*0.5)
        #     action[1] = 0.15*(random.random()-0.5)
        #     action[2] = action[2] * direction


        ret,pos = robot.get_tcp_position()
        # print(pos)
        cur_tcp_orn = euler_to_quaternion_rad(pos[3],pos[4],pos[5])

        encoded_action = encode_TCP_frame_actions(action,cur_tcp_orn)
        print(encoded_action)
        real_action = scale_actions(encoded_action) 
        # print(real_action)
        real_action = [real_action[0]*-11.5,real_action[1]*-11.5,real_action[2]*-11.5,real_action[3]*0.016,real_action[4]*0.016,real_action[5]*-0.016]
        time2 = time.time()

        time_wait = 0.1 - (time2-time1)

        if time_wait > 0:
            time.sleep(time_wait)
        else:
            pass

        if cv2.waitKey(5) & 0xFF == ord( 'q' ) :
            break


    time.sleep(0.1)    
    f.close()    
    cv2.destroyAllWindows()
    time.sleep(0.1)
    robot.servo_move_enable(Disable)
    time.sleep(0.1)
    robot.linear_move(init_pos,ABS,True,10)
    time.sleep(0.1)
    robot.motion_abort()
    # load_data(data_file_path, batch_size, img_size)



# [8.56, -0.01, ]

# [-370.0, -1.7316009737200488e-14, 79.99999999999997, 0.0, -0.0, 0.0]

# [-370.0, -1.7316009737200488e-14, 79.99999999999997, 0.0, -0.0, 0.0]

# [-370.09737529406556, 6.19868250868634e-05, 80.00007759864184, -1.1273757671954333e-07, -0.00020279056902398418, -2.328695885233347e-08]

# [-370.3495314179835, 0.00039358746028558475, 80.00044856372682, -7.786030546772606e-07, -0.0007783789545980607, -1.7461733381510258e-07]

# [-370.7479947784228, 0.0010854556641471669, 80.00117553727219, -2.228433952926655e-06, -0.0017609092081107538, -5.170046678456278e-07]

# [-371.1691812950293, 0.001727600617120701, 80.00209552802588, -3.1866930089782663e-06, -0.002549055251547237, -6.791903629865037e-07]

# [-371.64915219985403, 0.0022375343059125724, 80.00284353056816, -4.641654784451246e-06, -0.0031641705491847827, -0.0007475137305835327]