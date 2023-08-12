import pyrealsense2 as rs
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
import cv2.aruco as aruco
from scipy import optimize
import os
import sys
sys.path.append(os.getcwd())
from Camera import Camera
from utils import get_rigid_transform,plane_norm
from scipy.spatial.transform import Rotation as R
  
mtx = np.array([[613.89953613,   0,         311.69946289],[  0,         613.8996582,  237.63970947],[  0,           0,           1        ]])
dist = np.array(([[0.028784404196194664, 0.7481844381754564, 0.0028707604214314336, -0.0032153215725527914, -3.1796489988923713]]))

M = None

if __name__ == "__main__":
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Start streaming
    pipeline.start(config)
    try:
        while True:
            observed_pts = []                   #观察到的点（像素到空间点的转换）
            observed_pix = []                   #观察到的像素
            robot_pts = []                      #aruco to robot

            #这里内参可以直接等于mtx
            cam_intrinsics = mtx      #获取相机内参矩阵
                # cam_intrinsics = mtx
            depth_scale = 0.00012498664727900177        #获取相机深度尺寸
            checkerboard_size = (3,3)
            refine_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)   #迭代！？
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            camera_depth_img = np.asanyarray(depth_frame.get_data())
            camera_depth_img = camera_depth_img.astype(float) * depth_scale 
            camera_color_img = np.asanyarray(color_frame.get_data())
            cv2.imshow('color',camera_color_img)

            h1, w1 = camera_color_img.shape[:2]     #获取彩色图片的高、宽，并且赋值给h1和w1
                # print(h1, w1)
                

                # 纠正畸变
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (h1, w1), 0, (h1, w1))
            dst1 = cv2.undistort(camera_color_img, mtx, dist, None, newcameramtx)

            frame=dst1

            #灰度化，检测aruco标签，
            gray_data = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)            #转换为灰度图
            cv2.imshow('gray', gray_data)
                #cv2.waitKey(1)
                # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # 开始检测aruco码
            aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
            parameters = aruco.DetectorParameters_create()
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_data, aruco_dict, parameters=parameters) #左上、右上、右下、左下

            if ids is not None: #检测到角点
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.04, mtx, dist)  #角点坐标，aruco码尺寸单位m，内参矩阵，畸变参数；返回：旋转向量，平移向量
                rot_mat, _ = cv2.Rodrigues(rvec)
                tvec1 = np.reshape(tvec,(3,1))
                M = np.append(rot_mat, tvec1, axis=1)
                E = np.asarray([[0,0,0,1]])
                M = np.append(M,E,axis=0)
                # print(tvec)
                # print(M)
                # print(tvec1)
                # print(f"rot_mat:\n {rot_mat}")

            #     corners1 = np.reshape(corners[0], (4,2))    #把矩阵corners[0]变成4行2列

            #     #码的中点坐标（相机or像素！！！！）应该是像素吧
            #     x_pic = (corners1[1][0]+corners1[3][0])/2   
            #     y_pic = (corners1[0][1]+corners1[2][1])/2
            #     checkerboard_pix = np.array([x_pic, y_pic])                                               #找得到内点，就进一步精细化寻找

            #     checkerboard_pix=checkerboard_pix.astype(int)     
            #     checkerboard_z = camera_depth_img[checkerboard_pix[1]][checkerboard_pix[0]]         #证明了camera_depth_img应该是与camera_color_img是匹配对应的关系（如果不是很匹配呢）
            #     checkerboard_x = np.multiply(checkerboard_pix[0]-cam_intrinsics[0][2],checkerboard_z/cam_intrinsics[0][0])      #从2D像素转换为camera坐标系下的3D
            #     checkerboard_y = np.multiply(checkerboard_pix[1]-cam_intrinsics[1][2],checkerboard_z/cam_intrinsics[1][1])

            #     observed_pts.append([checkerboard_x,checkerboard_y,checkerboard_z])
            #     observed_pix.append(checkerboard_pix)   #observed_pix是相机坐标系
                for i in range(rvec.shape[0]):
                    cv2.drawFrameAxes(camera_color_img, mtx, dist, rvec[i, :, :], tvec[i, :, :], 0.03)
                    aruco.drawDetectedMarkers(camera_color_img, corners)
                cv2.imshow("frame", camera_color_img)
                cv2.waitKey(5)

            # observed_pts = np.asarray(observed_pts)                                                             #[4 5 6]]
            # observed_pix = np.asarray(observed_pix)

#             4.226425080103980714e-02 -7.354808316371801435e-01 6.762260564326902923e-01 -6.149212730621359535e-01
# -9.991051940046132840e-01 -3.003171264037147667e-02 2.978099308098344083e-02 -4.236163129845482711e-01
# -1.595122951464458469e-03 -6.768796366638366591e-01 -7.360919868145574529e-01 7.290960203739886847e-01
# 0.000000000000000000e+00 0.000000000000000000e+00 0.000000000000000000e+00 1.000000000000000000e+00

# 3.742053361891466839e-02 -7.364824511413819996e-01 6.754208338691214397e-01 -6.245209309594639269e-01
# -9.992637238225440255e-01 -3.330565112389221455e-02 1.904583564766280240e-02 -4.141999000371661133e-01
# 8.468406932826697334e-03 -6.756362429325386554e-01 -7.371865118950728935e-01 7.377924987108407384e-01
# 0.000000000000000000e+00 0.000000000000000000e+00 0.000000000000000000e+00 1.000000000000000000e+00

            camera2world = np.array([
                [ 9.927350783811934587e-01, -2.125771201391186277e-02, -1.184279267378213624e-01, -4.019149149378470343e-01],
[3.559344376017719441e-02, -8.883317404767283598e-01, 4.578207352478456116e-01, -2.863901985779893078e-01],
[-1.149355076239554629e-01, -4.587099612407824489e-01, -8.811214448336299743e-01, 5.870554887187741100e-01],
[0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]
])
            robot_pts = np.dot(camera2world,M)
            # print(np.shape(observed_pts))
            # print(robot_pts)

            obj_x = robot_pts[0][3]
            obj_y = robot_pts[1][3]
            r1 = robot_pts[0:3,0:3]
            r = R.from_matrix(r1)
            robot_rotate = np.asarray(r.as_euler('zyx')*180/3.1415926)
            obj_theta = robot_rotate[0]

            # desk_height = point_a[2] + (vector_norm[0]*(point_a[0]-obj_x)+vector_norm[1]*(point_a[1]-obj_y))/vector_norm[2]
            # calib_grid_z = calib_grid_z + desk_height

            obj = [1000*(obj_x-0.055), 1000*(obj_y+0.04), 1000*(robot_pts[2][3]-0.04), robot_rotate]
            print(obj)


            # checkerboard_offset_from_tool = [0.055,-0.04,0.040]
            # print(observed_pts)

    finally:
        # Stop streaming
        pipeline.stop()


            
