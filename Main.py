
from TaskController import TaskController
import sys
from PyQt5.QtWidgets import QApplication

if __name__ == '__main__':
    #这是主线程
    app = QApplication(sys.argv)
    
    task=TaskController()
    task.start_tasks()
    
    sys.exit(app.exec_())
   
   