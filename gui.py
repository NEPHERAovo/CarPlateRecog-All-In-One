from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from locate import locate
from keras import models
from cnn_reco import cnn_predict
import sys
import os
import cv2


class car_recog_gui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('车牌识别系统')
        self.setGeometry(round((1920-1024)/2), round((1080-768)/2), 1024, 768)
        print("loading...")
        self.unet = models.load_model('locate.h5')
        self.cnn = models.load_model('cnn.h5')
        print("finished")
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()

        # 原图显示
        ori_pic_layout = QVBoxLayout()
        ori_pic_label = QLabel()
        ori_pic_label.setText('原图:')
        self.ori_pic = QLabel()
        ori_pic_layout.addWidget(ori_pic_label)
        ori_pic_layout.addWidget(self.ori_pic)
        ori_pic_layout.addStretch(1)
        ori_pic_widget = QWidget()
        ori_pic_widget.setLayout(ori_pic_layout)

        main_layout.addWidget(ori_pic_widget)
        main_layout.addWidget(self.create_right_widget())

        main_widget = QWidget()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def create_right_widget(self):
        right_layout = QVBoxLayout()

        cut_pic_layout = QVBoxLayout()
        cut_pic_label = QLabel()
        cut_pic_label.setText('车牌区域:')
        self.cut_pic = QLabel()
        cut_pic_layout.addWidget(cut_pic_label)
        cut_pic_layout.addWidget(self.cut_pic)
        cut_pic_widget = QWidget()
        cut_pic_widget.setLayout(cut_pic_layout)
        right_layout.addWidget(cut_pic_widget)

        right_layout.addStretch(2)

        right_layout.addWidget(self.create_button())

        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        return right_widget

    def create_button(self):
        button_widget = QWidget()

        button_layout = QVBoxLayout()

        button_widget1 = QWidget()
        button_layout1 = QHBoxLayout()
        reco_button = QPushButton()
        reco_button.setText("单张识别")
        reco_button.clicked.connect(lambda: self.open_img())
        reco_all_button = QPushButton()
        reco_all_button.setText("集中识别")
        button_layout1.addWidget(reco_button)
        button_layout1.addWidget(reco_all_button)
        button_widget1.setLayout(button_layout1)

        button_widget2 = QWidget()
        button_layout2 = QHBoxLayout()
        up_button = QPushButton()
        up_button.setText("上一张")
        up_button.clicked.connect(lambda: self.pic_up())
        down_button = QPushButton()
        down_button.setText("下一张")
        down_button.clicked.connect(lambda: self.pic_down())
        button_layout2.addWidget(up_button)
        button_layout2.addWidget(down_button)
        button_widget2.setLayout(button_layout2)

        button_layout.addWidget(button_widget1)
        button_layout.addWidget(button_widget2)

        button_widget.setLayout(button_layout)

        return button_widget

    def predict_img(self):
        img = cv2.imread(self.path)
        img_cut, img_label = locate(img, self.unet)

        height, width, channel = img_cut.shape
        bytesPerLine = 3 * width
        qImg = QImage(img_cut.data, width, height,
                      bytesPerLine, QImage.Format_BGR888)

        height, width, channel = img_label.shape
        bytesPerLine = 3 * width
        qImg2 = QImage(img_label.data, width, height,
                       bytesPerLine, QImage.Format_BGR888)
        self.cut_pic.setPixmap(QPixmap(qImg).scaled(
            240, 80, Qt.KeepAspectRatio))
        self.ori_pic.setPixmap(QPixmap(qImg2).scaled(
            373, 600))
        print(cnn_predict(self.cnn, img_cut)[1])

    def open_img(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "open file", "", "ALL FILES (*.*);;PHOTO FILES (*.jpg)")

        if path:
            # self.ori_pic.setPixmap(QPixmap(path).scaled(
            #     600, 600, Qt.KeepAspectRatio))
            self.path = path
            self.predict_img()

    def pic_up(self):
        try:
            path = os.path.split(self.path)
            picName = os.listdir(path[0])
            n = picName.index(path[1])
            n = n-1
            if n < 0:
                n = len(picName) - 1
            self.path = path[0] + '/' + picName[n]
            if picName[n].split('.')[1] == 'jpg':
                # self.ori_pic.setPixmap(QPixmap(self.path).scaled(
                #     600, 600, Qt.KeepAspectRatio))
                self.predict_img()
            else:
                self.pic_up()
        except:
            msg = QMessageBox(QMessageBox.Critical, '错误', '未读入图片')
            msg.exec_()
            self.open_img()

    def pic_down(self):
        try:
            path = os.path.split(self.path)
            picName = os.listdir(path[0])
            n = picName.index(path[1])
            n = n+1
            if n >= len(picName):
                n = 0
            self.path = path[0] + '/' + picName[n]
            if picName[n].split('.')[1] == 'jpg':
                # self.ori_pic.setPixmap(QPixmap(self.path).scaled(
                #     600, 600, Qt.KeepAspectRatio))
                self.predict_img()
            else:
                self.pic_down()
        except:
            msg = QMessageBox(QMessageBox.Critical, '错误', '未读入图片')
            msg.exec_()
            self.open_img()


if __name__ == "__main__":
    app = QApplication([])
    window = car_recog_gui()
    window.show()
    sys.exit(app.exec())
