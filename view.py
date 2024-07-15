from common.Observe import Observable
from model.QRmodel import QRCodeDetector
from model.object_detect import ObjectDetector
from common.ThreadSafe import ThreadSafeObject
import cv2
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel













from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
import numpy as np
import copy
from sklearn.linear_model import LinearRegression
from pyorbbecsdk import *
import yaml
import time
import os
import sys
from utils import frame_to_bgr_image


MIN_DEPTH = 20  # 20mm
MAX_DEPTH = 10000  # 10000mm
font = cv2.FONT_HERSHEY_SIMPLEX

class TemporalFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.previous_frame = None

    def process(self, frame):
        if self.previous_frame is None:
            result = frame
        else:
            result = cv2.addWeighted(frame, self.alpha, self.previous_frame, 1 - self.alpha, 0)
        self.previous_frame = result
        return result
    
class CameraApp(QWidget,Observable,ThreadSafeObject):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.capture_video()
    
        self.ObjectDetector = ObjectDetector()
        
        self.ui_frame=None
        self.data=None
        
        self.R_cam2gripper = np.load('R_cam2gripper.npy')
        self.T_cam2gripper = np.load('T_cam2gripper.npy')
        
    def initUI(self):
        self.setWindowTitle('实时摄像头显示')
        self.setGeometry(100, 100, 640, 480)

        layout = QVBoxLayout()

        self.label = QLabel(self)
        layout.addWidget(self.label)

        self.setLayout(layout)

    def capture_video(self):
           # 创建管道
        self.pipeline = Pipeline()
        self.temporal_filter = TemporalFilter(alpha=0.5)
        config = Config()

        try:
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            #color_profile = profile_list.get_video_stream_profile(1280, 800, OBFormat.RGB, 30)
            color_profile = profile_list.get_default_video_stream_profile()
            config.enable_stream(color_profile)
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            #depth_profile = profile_list.get_video_stream_profile(1280, 800, OBFormat.Y16, 30)
            depth_profile = profile_list.get_default_video_stream_profile()
            config.enable_stream(depth_profile)
        except Exception as e:
            print(e)
            return

        try:
            self.pipeline.start(config)
        except Exception as e:
            print(e)
            return
       

        # 创建一个定时器，每秒更新一次摄像头帧
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000 / 30)  # 设置定时器，这里以30帧每秒的速度更新

    @ThreadSafeObject.thread_safe
    def update_frame(self):
        # 相机光心坐标和焦距
        cx, cy = 644.752, 395.306
        fx, fy = 611.963, 611.742

        # 获取摄像头帧
        depth_data = None
        color_image = None
        try:
            frames = self.pipeline.wait_for_frames(10)
            if frames is None:
                print('No frames received')
                return

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if color_frame:
                color_image = frame_to_bgr_image(color_frame)
            else:
                print('No color frame received')
                return

            if depth_frame:
                width, height = depth_frame.get_width(), depth_frame.get_height()
                scale = depth_frame.get_depth_scale()
                depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((height, width)).astype(np.float32) * scale
                depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0).astype(np.uint16)
                depth_data = self.temporal_filter.process(depth_data)

            # 目标检测
            frame, data = self.ObjectDetector.detect(color_image)
            
            self.data = copy.deepcopy(data)
            #print('data:', self.data)

            # 处理有效的z值
            valid_coordinates = []
            if depth_data is not None and self.data:
                for idx, item in enumerate(self.data):
                    x, y = item['Coordinate']
                    class_id = item['Property']['Color']
                    time.sleep(0.1)
                    z = depth_data[y, x] * scale
                    print(f"深度值: {z:.2f} mm",class_id)
                    #计算补偿z
                    
                    error_z = (x / 900) * 20
                    z=z-error_z
                    
                    if z == 0:
                        print(f"深度值无效，无法计算实际坐标（对象 {item['ObjectName']}）")
                        continue
                  
                    x_mm = (x - cx) * z / fx
                    y_mm = (y - cy) * z / fy
                    print(f"实际坐标: ({x_mm:.2f} mm, {y_mm:.2f} mm, {z:.2f} mm)")
                    
                   
                    x_mm, y_mm, z_m = self.trans_to_robot_coordinate(x_mm, y_mm, z)
                    # 更新坐标
                    new_item = item.copy()
                    z_offset = 90 if class_id in ['boxblue', 'boxgray'] else 1
                    new_item['Coordinate'] = [x_mm, y_mm-2, z_m + z_offset]
                    #new_item['Coordinate'] = [x_mm, y_mm, z_offset]
                    valid_coordinates.append(new_item)
                    
                    
                       # 计算文本显示的位置，稍微偏移对象的像素坐标
                    text_position_x = x + 10  # 在对象x坐标的基础上向右偏移10像素
                    text_position_y = y - 10  # 在对象y坐标的基础上向上偏移10像素
                    
                    # 确保文本位置在图像边界内
                    text_position_x = max(0, min(text_position_x, frame.shape[1] - 100))  # 假设文本宽度最多100像素
                    text_position_y = max(0, min(text_position_y, frame.shape[0] - 10))  # 假设文本高度最多10像素
                    
                    # 显示深度值和类别ID
                    text_to_display = f'Z: {z_m + z_offset:.2f} mm, Class: {class_id}'
                    cv2.putText(frame, text_to_display, (text_position_x, text_position_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    self.ui_frame = copy.deepcopy(frame)
            else:
                self.ui_frame = self.ui_frame = copy.deepcopy(frame)

            if valid_coordinates:
                self.data = valid_coordinates
            
            else:
                self.data = []
            print('data_send:', self.data)

            self.notify_observers()
            # 将 OpenCV 的 BGR 格式转换为 PyQt5 的 RGB 格式
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            

            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            p = convert_to_Qt_format.scaled(1280, 800, Qt.KeepAspectRatio)
            self.label.setPixmap(QPixmap.fromImage(p))

        except KeyboardInterrupt:
            print('KeyboardInterrupt')
            self.pipeline.stop()
 
    def trans_to_robot_coordinate(self,x,y,z):
        
        
        
        # 相机下物体的坐标（x,y,z,rx,ry,rz），机械臂末端基于基座的初始坐标（69, -277, 463,3.126,0.473,0.054），求出机械臂末端基于基座的位姿
        init_x=69
        init_y=-277
        init_z=463
        
        initial_gripper_pose = np.array([init_x, init_y, init_z], dtype=np.float64).reshape(3, 1)
        initial_gripper_orientation = np.array([3.126, 0.473, 0.054], dtype=np.float64)

        # 将初始旋转向量转换为旋转矩阵
        R_gripper2base_initial, _ = cv2.Rodrigues(initial_gripper_orientation)

        # 计算相机在基座坐标系下的旋转和平移向量
        R_cam2base = np.dot(R_gripper2base_initial, self.R_cam2gripper)
        T_cam2base = np.dot(R_gripper2base_initial, self.T_cam2gripper) + initial_gripper_pose

        # print("相机在基座坐标系下的旋转矩阵:")
        # print(R_cam2base)
        # print("相机在基座坐标系下的平移向量:")
        # print(T_cam2base)

        # 相机下物体的坐标（x, y, z, rx, ry, rz）
        object_in_camera = np.array([x, y, z], dtype=np.float64).reshape(3, 1)
        object_orientation_in_camera = np.array([0, 0, 0], dtype=np.float64)  # 假设物体在相机坐标系下没有旋转

        # 计算物体在基座坐标系下的坐标
        object_in_base = np.dot(R_cam2base, object_in_camera) + T_cam2base

        # 假设物体在相机坐标系下的旋转为单位矩阵
        R_object_in_camera, _ = cv2.Rodrigues(object_orientation_in_camera)
        R_object_in_base = np.dot(R_cam2base, R_object_in_camera)

        # 将旋转矩阵转换为旋转向量
        object_orientation_in_base, _ = cv2.Rodrigues(R_object_in_base)

        # print("物体在基座坐标系下的坐标:")
        # print(object_in_base)
        # print("物体在基座坐标系下的旋转向量:")
        # print(object_orientation_in_base)

        # 机械臂末端基于基座的抓取位姿
        gripper_pose_in_base = object_in_base
        gripper_orientation_in_base = object_orientation_in_base

        # print("机械臂末端基于基座的抓取位姿:")
        # print("位置:", gripper_pose_in_base)
        # print("方向:", gripper_orientation_in_base)
        
        robot_x,robot_y,robot_z = gripper_pose_in_base[0][0],gripper_pose_in_base[1][0],gripper_pose_in_base[2][0]
        return robot_x,robot_y,init_z-z
                
                
        
        
    def notify_observers(self):
        """Notifies all observing observers"""
        for observer in self._observers:
            observer.cameraObservable(self)
               

    def closeEvent(self, event):
        # 释放摄像头和关闭窗口
        self.cap.release()
        super().closeEvent(event)

if __name__ == '__main__':
    # 创建 QApplication 对象
    app = QApplication(sys.argv)
    ex = CameraApp()
    ex.show()
    
    sys.exit(app.exec_())
    
   
    
    
