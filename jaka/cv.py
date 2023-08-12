import pyrealsense2 as rs
import cv2
import numpy as np

# 初始化摄像头
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

try:
    while True:
        # 等待获取帧
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            continue
        
        # 将帧数据转换为OpenCV图像格式
        color_image = np.asanyarray(color_frame.get_data())
        
        # 显示图像
        cv2.imshow('RealSense Camera', color_image)
        
        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
