import cv2
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import  QWidget, QLabel, QApplication
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

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


class App(QWidget):
    def __init__(self):
            super(App,self).__init__()
            self.title = 'Deteksi Kematangan Strowberry'
            self.left = 100
            self.top = 100
            self.width = 640
            self.height = 480
            self.initUI()

    def initUI(self):
            self.setWindowTitle(self.title)
            self.setGeometry(self.left, self.top, self.width, self.height)
            self.resize(660, 581)
            self.gridLayoutWidget = QWidget()
            self.gridLayoutWidget.setGeometry(QtCore.QRect(10, 10, 641, 21))
            self.gridLayoutWidget.setObjectName("gridLayoutWidget")
            self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
            self.gridLayout.setContentsMargins(0, 0, 0, 0)
            self.gridLayout.setSpacing(13)
            self.gridLayout.setObjectName("gridLayout")
            self.label.setObjectName("label")
            self.label.setText("Input Gambar")
            self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
            self.label = QtWidgets.QLabel(self.gridLayoutWidget)
            self.label.setAlignment(QtCore.Qt.AlignCenter)
            self.label = QLabel(self)
            self.label.move(0, 0)
            self.label.resize(640, 480)
            th = Thread(self)
            th.changePixmap.connect(lambda p: self.setPixMap(p))
            th.start()

    def setPixMap(self, p):     
        p = QPixmap.fromImage(p)    
        p = p.scaled(640, 480, Qt.KeepAspectRatio)
        self.label.setPixmap(p)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())