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

class CameraApp(QWidget,Observable,ThreadSafeObject):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.capture_video()

        self.ObjectDetector = ObjectDetector()
        
        self.ui_frame=None
        self.data=None
        
    def initUI(self):
        self.setWindowTitle('实时摄像头显示')
        self.setGeometry(100, 100, 640, 480)

        layout = QVBoxLayout()

        self.label = QLabel(self)
        layout.addWidget(self.label)

        self.setLayout(layout)

    def capture_video(self):
        # 使用 OpenCV 打开摄像头，0 通常代表第一个摄像头
        self.cap = cv2.VideoCapture(2)

        # 创建一个定时器，每秒更新一次摄像头帧
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000 / 30)  # 设置定时器，这里以30帧每秒的速度更新

    @ThreadSafeObject.thread_safe
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            #frame=cv2.resize(frame,(1024,576))
            frame=cv2.resize(frame,(640,480))
            #目标检测
            frame,data=self.ObjectDetector.detect(frame)
            #复制图像
            self.ui_frame=copy.deepcopy(frame)
            self.data=copy.deepcopy(data)
            
            self.notify_observers()
 
            # 将 OpenCV 的 BGR 格式转换为 PyQt5 的 RGB 格式
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            p = convert_to_Qt_format.scaled(1024, 576, Qt.KeepAspectRatio)
            # 设置 QLabel 显示图片
            self.label.setPixmap(QPixmap.fromImage(p))

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
    
   
    
    
