# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Stroberi.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(751, 423)
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


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())