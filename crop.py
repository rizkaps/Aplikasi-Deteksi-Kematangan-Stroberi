import sys
from PyQt5 import QtCore, QtGui, QtWidgets

class Crop(QtWidgets.QLabel):
    def __init__(self, parentQWidget = None):
        super(Crop, self).__init__(parentQWidget)
        self.initUI()
        self.w = 0
        self.h = 0
        self.disW = 0
        self.disH = 0
        self.skala = 0 
        # self.adjustSize()

    def initUI (self):
        self.setWindowTitle(("Croping Gambar"))
        self.setWindowIcon(QtGui.QIcon('assets/appicon.jpg'))
        pix = QtGui.QPixmap('assets/input/input.jpg')
        self.w = pix.width()
        self.h = pix.height()
        if self.w > 1300:
            self.disW = self.w/5
            self.disH = self.h/5
            self.skala = 5
        else:
            self.disW = self.w/3
            self.disH = self.h/3
            self.skala = 3
        scaledimage = pix.scaled(self.disW,self.disH,QtCore.Qt.KeepAspectRatio)
        self.setPixmap(QtGui.QPixmap(scaledimage))
        self.setGeometry(100, 100, self.disW, self.disH)

    def mousePressEvent (self, eventQMouseEvent):
        self.originQPoint = eventQMouseEvent.pos()
        self.currentQRubberBand = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self)
        self.currentQRubberBand.setGeometry(QtCore.QRect(self.originQPoint, QtCore.QSize()))
        # print(self.originQPoint)
        self.currentQRubberBand.show()

    def mouseMoveEvent (self, eventQMouseEvent):
        self.currentQRubberBand.setGeometry(QtCore.QRect(self.originQPoint, eventQMouseEvent.pos()).normalized())
        # print(self.originQPoint)

    def mouseReleaseEvent (self, eventQMouseEvent):
        self.currentQRubberBand.hide()
        tempCurrentQRect = self.currentQRubberBand.geometry()
        print(tempCurrentQRect)
        self.currentQRubberBand.deleteLater()
        cropQPixmap = self.pixmap().copy(tempCurrentQRect)
        cropQPixmap.save('assets/output/output.jpg')
        self.close()

if __name__ == '__main__':
    myQApplication = QtWidgets.QApplication(sys.argv)
    myQExampleLabel = Crop()
    myQExampleLabel.show()
    sys.exit(myQApplication.exec_())