from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from crop import Crop
import shutil
import sys
import os
import os.path
from os import path
import time
import numpy
import cv2
import sklearn
import pandas as pd
import numpy as np
import threading
from matplotlib import pyplot as pyplot
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

class Thread(QThread):
    changePixmap = pyqtSignal(QImage)

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret:
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)

class Main(object):
    def __init__(self, Form):
        self.form = Form
        self.form.setObjectName("Form")
        self.form.resize(751, 423)
        self.form.setWindowIcon(QtGui.QIcon('assets/appicon.jpg')) 
        self.groupBox = QtWidgets.QGroupBox(Form)
        self.groupBox.setGeometry(QtCore.QRect(0, 310, 631, 101))
        self.groupBox.setObjectName("groupBox")
        self.btDeteksi = QtWidgets.QPushButton(self.groupBox)
        self.btDeteksi.setGeometry(QtCore.QRect(10, 20, 91, 71))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.btDeteksi.setFont(font)
        self.btDeteksi.setObjectName("btDeteksi")
        self.gridLayoutWidget = QtWidgets.QWidget(self.groupBox)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(110, 10, 511, 80))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_17 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_17.setObjectName("label_17")
        self.gridLayout.addWidget(self.label_17, 2, 2, 1, 1)
        self.label_15 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_15.setObjectName("label_15")
        self.gridLayout.addWidget(self.label_15, 4, 0, 1, 1)
        self.lHue = QtWidgets.QLabel(self.gridLayoutWidget)
        self.lHue.setText("")
        self.lHue.setObjectName("lHue")
        self.gridLayout.addWidget(self.lHue, 2, 3, 1, 1)
        self.lBlue = QtWidgets.QLabel(self.gridLayoutWidget)
        self.lBlue.setText("")
        self.lBlue.setObjectName("lBlue")
        self.gridLayout.addWidget(self.lBlue, 4, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 3, 0, 1, 1)
        self.lRed = QtWidgets.QLabel(self.gridLayoutWidget)
        self.lRed.setText("")
        self.lRed.setObjectName("lRed")
        self.gridLayout.addWidget(self.lRed, 2, 1, 1, 1)
        self.lGreen = QtWidgets.QLabel(self.gridLayoutWidget)
        self.lGreen.setText("")
        self.lGreen.setObjectName("lGreen")
        self.gridLayout.addWidget(self.lGreen, 3, 1, 1, 1)
        self.label_18 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_18.setObjectName("label_18")
        self.gridLayout.addWidget(self.label_18, 3, 2, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 1, 0, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 2, 0, 1, 1)
        self.lDiameter = QtWidgets.QLabel(self.gridLayoutWidget)
        self.lDiameter.setText("")
        self.lDiameter.setObjectName("lDiameter")
        self.gridLayout.addWidget(self.lDiameter, 2, 5, 1, 1)
        self.label_19 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_19.setObjectName("label_19")
        self.gridLayout.addWidget(self.label_19, 1, 5, 1, 1)
        self.lSarutation = QtWidgets.QLabel(self.gridLayoutWidget)
        self.lSarutation.setText("")
        self.lSarutation.setObjectName("lSarutation")
        self.gridLayout.addWidget(self.lSarutation, 3, 3, 1, 1)
        self.label_24 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_24.setObjectName("label_24")
        self.gridLayout.addWidget(self.label_24, 4, 2, 1, 1)
        self.lValue = QtWidgets.QLabel(self.gridLayoutWidget)
        self.lValue.setText("")
        self.lValue.setObjectName("lValue")
        self.gridLayout.addWidget(self.lValue, 4, 3, 1, 1)
        self.groupBox_2 = QtWidgets.QGroupBox(Form)
        self.groupBox_2.setGeometry(QtCore.QRect(640, 310, 111, 101))
        self.groupBox_2.setStyleSheet("")
        self.groupBox_2.setObjectName("groupBox_2")
        self.lHasil = QtWidgets.QLabel(self.groupBox_2)
        self.lHasil.setGeometry(QtCore.QRect(10, 20, 101, 71))
        font = QtGui.QFont()
        font.setPointSize(48)
        self.lHasil.setFont(font)
        self.lHasil.setObjectName("lHasil")
        self.gridLayoutWidget_2 = QtWidgets.QWidget(Form)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(10, 12, 731, 289))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.lTest = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.lTest.setMinimumSize(QtCore.QSize(362, 220))
        self.lTest.setMaximumSize(QtCore.QSize(362, 220))
        self.lTest.setAutoFillBackground(False)
        self.lTest.setStyleSheet("background-color: white")
        self.lTest.setFrameShape(QtWidgets.QFrame.Box)
        self.lTest.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.lTest.setLineWidth(2)
        self.lTest.setText("")
        self.lTest.setPixmap(QtGui.QPixmap("assets/iconLoad.png"))
        self.lTest.setScaledContents(False)
        self.lTest.setAlignment(QtCore.Qt.AlignCenter)
        self.lTest.setObjectName("lTest")
        self.gridLayout_2.addWidget(self.lTest, 1, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)
        self.lCrop = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.lCrop.setMinimumSize(QtCore.QSize(361, 220))
        self.lCrop.setMaximumSize(QtCore.QSize(361, 220))
        self.lCrop.setAutoFillBackground(False)
        self.lCrop.setStyleSheet("background-color: white")
        self.lCrop.setFrameShape(QtWidgets.QFrame.Box)
        self.lCrop.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.lCrop.setLineWidth(2)
        self.lCrop.setText("")
        self.lCrop.setPixmap(QtGui.QPixmap("assets/crop.png"))
        self.lCrop.setScaledContents(False)
        self.lCrop.setAlignment(QtCore.Qt.AlignCenter)
        self.lCrop.setObjectName("lCrop")
        self.gridLayout_2.addWidget(self.lCrop, 1, 1, 1, 1)
        self.btPilih = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.btPilih.setObjectName("btPilih")
        self.gridLayout_2.addWidget(self.btPilih, 2, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.gridLayout_2.addWidget(self.label_2, 0, 1, 1, 1)
        self.btCrop = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.btCrop.setObjectName("btCrop")
        self.gridLayout_2.addWidget(self.btCrop, 2, 1, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)


        self.imgPath = ""
        self.imgHeight = 0
        self.imgWidth = 0
        self.channels = 0
        self.btPilih.clicked.connect(lambda: self.getImage())
        self.btCrop.clicked.connect(lambda: self.getCrop())
        self.btDeteksi.clicked.connect(lambda: self.Prediksi())
        # self.th = Thread(self)
        # self.th.changePixmap.connect(lambda p: self.setPixMap(p))
        # self.th.start()
        

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Deteksi Kematangan Stroberi"))
        self.groupBox.setTitle(_translate("Form", "Deteksi Kematangan"))
        self.btDeteksi.setText(_translate("Form", "Deteksi"))
        self.label_17.setText(_translate("Form", "Hue"))
        self.label_15.setText(_translate("Form", "Blue"))
        self.label_3.setText(_translate("Form", "Green"))
        self.label_18.setText(_translate("Form", "Sarutation"))
        self.label_7.setText(_translate("Form", "Nilai Gambar"))
        self.label_8.setText(_translate("Form", "Red"))
        self.label_19.setText(_translate("Form", "Size"))
        self.label_24.setText(_translate("Form", "Value"))
        self.groupBox_2.setTitle(_translate("Form", "Hasil Deteksi"))
        self.lHasil.setText(_translate("Form", "- -"))
        self.label.setText(_translate("Form", "Input Gambar"))
        self.btPilih.setText(_translate("Form", "Pilih Gambar"))
        self.label_2.setText(_translate("Form", "Crop Gambar"))
        self.btCrop.setText(_translate("Form", "Crop"))

    def setPixMap(self, p):     
        p = QPixmap.fromImage(p)    
        p = p.scaled(640, 480, Qt.KeepAspectRatio)
        self.lTest.setPixmap(p)

    def getImage(self):
        self.imgPath = QtWidgets.QFileDialog.getOpenFileName(
            Form, 'Open Image',os.getcwd(), "Image files (*.jpg *.png) ")
        if self.imgPath[0] == "":
            pixmap = QPixmap("assets/iconLoad.png")
            self.lTest.setPixmap(QPixmap(pixmap))
        else:
            pixmap = QPixmap(self.imgPath[0])
            self.lTest.setPixmap(QPixmap(pixmap))
            self.lTest.setScaledContents(True)
            # img = cv2.imread(self.imgPath[0])
            # self.imgHeight, self.imgWidth, self.channels = img.shape
            # print(self.imgHeight, self.imgWidth, self.channels)
            shutil.copyfile(self.imgPath[0], 'assets/input/input.jpg')
    
    def getCrop(self):
        # if path.exists('assets/output/output.jpg'):
        self._crop = Crop()
        self._crop.show()
        pixmap = QPixmap('assets/output/output.jpg')
        self.lCrop.setPixmap(QPixmap(pixmap))
        self.lCrop.setScaledContents(True)

    def Prediksi(self):
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
        self.lRed.setText(str(r))
        self.lGreen.setText(str(g))
        self.lBlue.setText(str(b))
        self.lHue.setText(str(h))
        self.lSarutation.setText(str(s))
        self.lValue.setText(str(v))
        self.lDiameter.setText(str(white_pix))
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

        Form = QtWidgets.QWidget()
        main = Main(Form)
        Form.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(e)
