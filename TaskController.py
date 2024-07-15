from common.Observe import Observer
from PyQt5.QtWidgets import QApplication
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import sys
from view import CameraApp 
from model.socket_connect import SocketServer
from model.socket_ui import SocketUi
import cv2

class TaskController(Observer):
    def __init__(self):
        super().__init__()
        self.pool = ThreadPoolExecutor(max_workers=10)
        self.app = QApplication(sys.argv)  # 在这里创建 QApplication 实例
       
        self.__camera = CameraApp()
        self.__camera.show()
        self.__camera.register_observer(self)
        
        self.__socket=SocketServer()
        self.__socket.register_observer(self)
        
        self.__socket_ui=SocketUi()
        self.__socket_ui.register_observer(self)
        
        print('初始化任务控制器完成')

    
    def cameraObservable(self,observerable):
        
        assert isinstance(observerable, CameraApp)  
        self.__socket_ui.message=self.__camera.ui_frame
        self.__socket.message=self.__camera.data
        
        
    def socketObservable(self,observerable):
        assert isinstance(observerable, SocketServer)  
        pass
    
    def uiSocketObservable(self,observerable):
        assert isinstance(observerable, SocketUi) 
        print('prompt:',self.__socket_ui.mico_data)
        self.__socket.mic_data=self.__socket_ui.mico_data

    def run_socket(self):
        self.__socket.start_server()

    def run_socket_ui(self):
        self.__socket_ui.start_ui_server()

    #开启多线程
    def start_tasks(self):
        
        self.pool.submit(self.run_socket)
        self.pool.submit(self.run_socket_ui)
        print('开启多线程完成')
        

    def shutdown_pool(self):
        self.pool.shutdown(wait=True)
        print("ThreadPoolExecutor has been shut down.")
        
