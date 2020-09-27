import os
import sys
import shutil
import os.path
from os import path
import time
import numpy
import cv2
import sklearn
import pandas as pd
import numpy as np
import threading
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import  QWidget, QLabel, QApplication
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from matplotlib import pyplot as pyplot
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

class Thread(QThread):
    changePixmap = pyqtSignal(QImage)
    
    def __init__(self, status, capture, parent=None):
        QThread.__init__(self, parent)
        self.status = status
        self.capture = capture

    
    def run(self):
        self.cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = self.cap.read()
            if ret:
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # hsv = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2HSV)
                # kernal = numpy.ones((5, 5), "uint8")
                # lower_red1 = numpy.array([0,120,70])
                # upper_red1 = numpy.array([10,255,255])
                # mask = cv2.inRange(hsv,lower_red1,upper_red1)
                # lower_red2 = numpy.array([170,120,70])
                # upper_red2 = numpy.array([180,255,255])
                # mask1 = cv2.inRange(hsv,lower_red2,upper_red2)
                # mask = mask+mask1
                # mask = cv2.dilate(mask, kernal)
                # res_red = cv2.bitwise_and(rgbImage, rgbImage,mask = mask)
                # contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # for pic, contour in enumerate(contours):
                #     area = cv2.contourArea(contour)
                #     if(area > 100):
                #         x, y, w, h = cv2.boundingRect(contour)
                #         rgbImage = cv2.rectangle(rgbImage, (x, y),(x + w, y + h),(0, 0, 255), 2)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)
            if self.capture == True:
                cv2.imwrite('assets/input/input.jpg', frame)
            if self.status == False:
                break
        self.cap.release()
        cv2.destroyAllWindows()
    

class Main(QWidget):
    def __init__(self):
        super(QWidget,self).__init__()
        self.setObjectName("Form")
        self.resize(660, 581)
        self.setWindowIcon(QtGui.QIcon('assets/appicon.jpg'))
        self.gridLayoutWidget = QtWidgets.QWidget(self)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(10, 10, 641, 21))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(13)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.lGambar = QtWidgets.QLabel(self)
        self.lGambar.setGeometry(QtCore.QRect(10, 40, 640, 420))
        self.lGambar.setMinimumSize(QtCore.QSize(362, 220))
        self.lGambar.setAutoFillBackground(False)
        self.lGambar.setStyleSheet("background-color: white")
        self.lGambar.setFrameShape(QtWidgets.QFrame.Box)
        self.lGambar.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.lGambar.setLineWidth(2)
        self.lGambar.setText("")
        self.lGambar.setPixmap(QtGui.QPixmap("assets/iconLoad.png"))
        self.lGambar.setScaledContents(False)
        self.lGambar.setAlignment(QtCore.Qt.AlignCenter)
        self.lGambar.setObjectName("lGambar")
        self.gridLayoutWidget_2 = QtWidgets.QWidget(self)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(10, 470, 641, 101))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.groupBox_3 = QtWidgets.QGroupBox(self.gridLayoutWidget_2)
        self.groupBox_3.setObjectName("groupBox_3")
        self.btPilih = QtWidgets.QPushButton(self.groupBox_3)
        self.btPilih.setGeometry(QtCore.QRect(10, 20, 191, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.btPilih.setFont(font)
        self.btPilih.setObjectName("btPilih")
        self.btStream = QtWidgets.QPushButton(self.groupBox_3)
        self.btStream.setGeometry(QtCore.QRect(10, 60, 191, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.btStream.setFont(font)
        self.btStream.setObjectName("btStream")
        self.gridLayout_2.addWidget(self.groupBox_3, 0, 0, 1, 1)
        self.groupBox = QtWidgets.QGroupBox(self.gridLayoutWidget_2)
        self.groupBox.setObjectName("groupBox")
        self.btDeteksi = QtWidgets.QPushButton(self.groupBox)
        self.btDeteksi.setGeometry(QtCore.QRect(10, 20, 191, 71))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.btDeteksi.setFont(font)
        self.btDeteksi.setObjectName("btDeteksi")
        self.gridLayout_2.addWidget(self.groupBox, 0, 1, 1, 1)
        self.groupBox_2 = QtWidgets.QGroupBox(self.gridLayoutWidget_2)
        self.groupBox_2.setStyleSheet("")
        self.groupBox_2.setObjectName("groupBox_2")
        self.lHasil = QtWidgets.QLabel(self.groupBox_2)
        self.lHasil.setGeometry(QtCore.QRect(10, 20, 101, 71))
        font = QtGui.QFont()
        font.setPointSize(48)
        self.lHasil.setFont(font)
        self.lHasil.setObjectName("lHasil")
        self.gridLayout_2.addWidget(self.groupBox_2, 0, 2, 1, 1)

        self.retranslateUi(QWidget)
        QtCore.QMetaObject.connectSlotsByName(self)

        self.imgPath = ""
        self.cam_status = False
        self.cam_capture = False
        self.Camera_thread = Thread(status=self.cam_status, capture=self.cam_capture)
        self.btStream.clicked.connect(lambda: self.startCamera())
        self.btPilih.clicked.connect(lambda: self.getImage())
        self.btDeteksi.clicked.connect(lambda: self.Prediksi())

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("Form", "Deteksi Kematangan Strowberry"))
        self.label.setText(_translate("Form", "Input Gambar"))
        self.groupBox_3.setTitle(_translate("Form", "Deteksi Kematangan"))
        self.btPilih.setText(_translate("Form", "Pilih Gambar"))
        self.btStream.setText(_translate("Form", "Streaming Camera"))
        self.groupBox.setTitle(_translate("Form", "Deteksi Kematangan"))
        self.btDeteksi.setText(_translate("Form", "Deteksi"))
        self.groupBox_2.setTitle(_translate("Form", "Hasil Deteksi"))
        self.lHasil.setText(_translate("Form", "- -"))

    def startCamera(self):
        if self.cam_status == False:
            self.cam_status = True
            self.Camera_thread = Thread(status=self.cam_status, capture=self.cam_capture)
            self.Camera_thread.changePixmap.connect(lambda p: self.setPixMap(p))
            self.Camera_thread.start()
            self.btStream.setText("Stop Camera")
        else:
            self.cam_status = False
            self.cam_capture = False
            self.Camera_thread.changePixmap.connect(lambda: self.setPixDef())
            self.Camera_thread = Thread(status=self.cam_status, capture=self.cam_capture)
            self.Camera_thread.exit()
            self.btStream.setText("Streaming Camera")
            self.lGambar.setPixmap(QtGui.QPixmap("assets/iconLoad.png"))

    def setPixMap(self, p):     
        p = QPixmap.fromImage(p)    
        p = p.scaled(640, 480, Qt.KeepAspectRatio)
        self.lGambar.setPixmap(p)

    def setPixDef(self):
        self.lGambar.setPixmap(QtGui.QPixmap("assets/iconLoad.png"))
        
    def getImage(self):
        self.imgPath = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open Image',os.getcwd(), "Image files (*.jpg *.png) ")
        if self.imgPath[0] == "":
            pixmap = QPixmap("assets/iconLoad.png")
            self.lGambar.setPixmap(QPixmap(pixmap))
        else:
            shutil.copyfile(self.imgPath[0], 'assets/input/input.jpg')
            img = cv2.imread(self.imgPath[0])
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            kernal = numpy.ones((5, 5), "uint8")
            lower_red1 = numpy.array([0,120,70])
            upper_red1 = numpy.array([10,255,255])
            mask = cv2.inRange(hsv,lower_red1,upper_red1)
            lower_red2 = numpy.array([170,120,70])
            upper_red2 = numpy.array([180,255,255])
            mask1 = cv2.inRange(hsv,lower_red2,upper_red2)
            mask = mask+mask1
            mask = cv2.dilate(mask, kernal)
            res_red = cv2.bitwise_and(img, img,mask = mask)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            x, y, w, h = cv2.boundingRect(contours[0])
            roi = img[y:y+h, x:x+w]
            # print(len(contours))
            i = 0
            contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
            biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
            # print(biggest_contour)
            x, y, w, h = cv2.boundingRect(biggest_contour)
            img = cv2.rectangle(img, (x, y),(x + w, y + h),(0, 0, 255), 2)
            cv2.imwrite('assets/output/output.jpg', img)
            cv2.imwrite('assets/output/output2.jpg', roi)
            pixmap = QPixmap('assets/output/output.jpg')
            pixmap.scaled(640, 480, Qt.KeepAspectRatio)
            self.lGambar.setPixmap(QPixmap(pixmap))
            self.lGambar.setScaledContents(True)

    def Prediksi(self):
        if self.cam_status == True:
            self.cam_status = False
            self.cam_capture = True
            self.Camera_thread = Thread(status=self.cam_status, capture=self.cam_capture)
            self.Camera_thread.exit()
            self.btStream.setText("Streaming Camera")
            self.lGambar.setPixmap(QtGui.QPixmap("assets/input/input.jpg"))
            self.create_prediction()
        else:
            self.create_prediction()

    def create_prediction(self):
        img = cv2.imread('assets/input/input.jpg')
        imgRgb = img
        imgHsv = img
        R, G, B = cv2.split(imgRgb)
        arrR = numpy.array(R)
        arrG = numpy.array(G)
        arrB = numpy.array(B)
        r = numpy.mean(arrR)
        g = numpy.mean(arrG)
        b = numpy.mean(arrB)
        hsv = cv2.cvtColor(imgHsv,cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)
        arrH = numpy.array(H)
        arrS = numpy.array(S)
        arrV = numpy.array(V)
        h = numpy.mean(arrH)
        s = numpy.mean(arrS)
        v = numpy.mean(arrV)
        lower_red = numpy.array([0,120,70])
        upper_red = numpy.array([10,255,255])
        mask1 = cv2.inRange(hsv,lower_red,upper_red)
        lower_red = numpy.array([170,120,70])
        upper_red = numpy.array([180,255,255])
        mask2 = cv2.inRange(hsv,lower_red,upper_red)
        mask1 = mask1+mask2
        white_pix = numpy.sum(mask1 == 255)
        df = pd.read_csv('Dataset_extraction.csv')
        le = LabelEncoder()
        y = le.fit_transform(df['Label'])
        test = numpy.array([r,g,b,h,s,v,white_pix])
        matrix_W = numpy.load('matrix_W.npy')
        X_lda = numpy.load('X_lda.npy')
        print(test)
        print(matrix_W)
        test = numpy.array(test.dot(matrix_W))
        test = np.reshape(test, (-1, 2))
        X_train,X_test ,y_train, y_test = train_test_split(X_lda, y, random_state=111)
        Model1 = RandomForestClassifier()
        Model1.fit(X_train, y_train)
        y_pred = Model1.predict(test)
        print(y_pred)
        predict = ""
        if y_pred[0] == 0:
            predict = "A"
        elif y_pred[0] == 1:
            predict = "B"
        elif y_pred[0] == 2:
            predict = "C"
        self.lHasil.setText(predict)

if __name__ == "__main__":
    try:
        app = QtWidgets.QApplication(sys.argv)
        main = Main()
        main.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(e)