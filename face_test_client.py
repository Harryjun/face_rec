# face_rec_api by tcp
# Author: menglingjun@cloudcver.com
#

#package

import socket
import time
from PyQt5.QtWidgets import QWidget, QPushButton, QApplication,QLabel,QLineEdit,QMessageBox,QInputDialog,QTextEdit
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtCore import Qt, QThread,pyqtSignal
import sys

class DownloadThread(QThread):
    trigger = pyqtSignal(int,str)

    def __int__(self):
        super(DownloadThread, self).__init__()
    def set_port_url(self,port_default,url):
        self.port_default = port_default
        self.url = url
    def run(self):
        try:
            # 1.创建套接字
            tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            IP_default = '39.97.178.93'
            # 3.链接服务器
            tcp_socket.connect((IP_default, self.port_default))

            # 4.获取文件下载的文件名
            # download_file_name = input("请输入要下载的文件名字：")
            image_url = self.url
            # 'http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz'
            # 5.将文件名字发送到服务器

            tcp_socket.send(image_url.encode("utf-8"))
            print(4)
            # 6.接收文件中的数据
            class_image = tcp_socket.recv(1024).decode("utf-8")
            print(class_image)

            # 8.关闭套接字
            tcp_socket.close()
            if class_image != "ERROR":
                self.trigger.emit(1,str(class_image))
            else:
                self.trigger.emit(0, '-')
        except:
            print("下载失败了")
            self.trigger.emit(0,'-')


#http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz

## 窗口
class Example(QWidget):

    def __init__(self):
        super().__init__()
        self.ok = False
        self.port_default = 0
        self.content_tmp = ''
        self.initUI()
    #UI界面涉及
    def initUI(self):
        #1 说明内容
        self.lbl2 = QLabel(self)
        self.lbl2.setText(
            "1、You should input Port first.\n2、Enter the URL and click the rec button\n3、If you have some problem, send email to\n menglingjun@cloudcver.com")
        self.lbl2.move(30, 10)
        #2 下载区
        #2.1 文本URL
        self.lbl = QLabel(self)
        self.lbl.setText("Url:")
        self.lbl.move(30, 93)
        #2.2 链接URL
        self.file_url = QLineEdit(self)
        self.file_url.move(50, 90)
        self.file_url.setGeometry(50, 90, 250, 23)
        #2.3 下载按钮
        download_button = QPushButton('rec', self)
        download_button.clicked.connect(self.download)
        download_button.move(320, 90)
        #3 端口号按钮
        self.download_button = QPushButton('No Port', self)
        self.download_button.clicked.connect(self.login)
        self.download_button.move(400, 10)
        #4 日志
        self.tmp = QTextEdit(self)
        self.tmp.move(50, 120)
        self.tmp.setGeometry(50, 120, 350, 70)

        #5 标题
        self.setGeometry(300, 300, 500, 200)
        self.setWindowTitle('274类政治人物识别V1.2@cloudcver')
        self.show()
    def login(self):
        text, ok = QInputDialog.getText(self, 'Set Port',
                                        'Enter your port:')
        if ok:
            self.port_default = int(text)
            self.download_button.setText(text)
            self.content_tmp = "%s:Set port at %s\n"%(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),text)  + self.content_tmp
            self.tmp.setText(self.content_tmp)
            self.ok = True
    def download(self):
        if self.ok:
            self.content_tmp = "%s:Send the task to server\n" % (
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))  + self.content_tmp
            self.tmp.setText(self.content_tmp)
            self.thread = DownloadThread()
            self.thread.set_port_url(self.port_default,self.file_url.text())  # 创建
            self.thread.trigger.connect(self.finish)  # 连接信号
            self.thread.start()  # 开始线程
        else:
            reply = QMessageBox.question(self, 'Message',
                                         "You should set port first", QMessageBox.Yes)
    def finish(self,status,class_image):
        print("sdsd")
        if status == 1:
            self.content_tmp = "%s:Finish! The images is  %s\n" % ((time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),class_image) + self.content_tmp
        else:
            self.content_tmp = "%s:Fail!\n" % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) + self.content_tmp
        self.tmp.setText(self.content_tmp)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())