'''
created on 06/20/2020
author: 杨宇轩，钟余盛
description: 界面，包含各种识别方式
'''

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from locate import locate
from keras import models
from cnn_reco import cnn_predict
from detect_yolo import detect
from pathlib import Path
import sys
import os
import cv2
import time
import qieGeHanShu as qgHS
import ZhiFuShiBie as ZF

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
sys.path.append(str(ROOT) + '\\yolov5')  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov5.utils.torch_utils import select_device
from yolov5.models.common import DetectMultiBackend


class car_recog_gui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('车牌识别系统')
        self.setGeometry(round((1920 - 1024) / 2), round((1080 - 768) / 2),
                         1024, 768)
        print("loading...")
        self.way = None
        self.unet = models.load_model('weights/locate.h5')
        self.cnn = models.load_model('weights/cnn.h5')
        self.device = select_device('0')
        self.text = ''
        self.yolo = DetectMultiBackend('weights/yolo.pt',
                                       device=self.device,
                                       data=ROOT / 'scripts/CCPD.yaml')
        print("finished")
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()

        # 原图显示
        ori_pic_layout = QVBoxLayout()
        ori_pic_label = QLabel()
        ori_pic_label.setText('原图:')
        self.ori_pic = QLabel()
        self.ori_pic.setFixedSize(373, 600)
        ori_pic_layout.addWidget(ori_pic_label)
        ori_pic_layout.addWidget(self.ori_pic)
        ori_pic_layout.addStretch(1)
        ori_pic_widget = QWidget()
        ori_pic_widget.setLayout(ori_pic_layout)

        dialog_label = QLabel()
        dialog_label.setText('log:')
        self.dialog = QTextEdit()
        self.dialog.setReadOnly(True)
        self.dialog.setMinimumHeight(600)
        self.dialog.setMaximumWidth(240)
        dialog_layout = QVBoxLayout()
        dialog_layout.addWidget(dialog_label)
        dialog_layout.addWidget(self.dialog)
        dialog_layout.addStretch(1)
        dialog_widget = QWidget()
        dialog_widget.setLayout(dialog_layout)

        main_layout.addWidget(ori_pic_widget)
        main_layout.addStretch(1)
        main_layout.addWidget(dialog_widget)

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
        self.cut_pic.setFixedSize(240, 80)
        cut_pic_layout.addWidget(cut_pic_label)
        cut_pic_layout.addWidget(self.cut_pic)
        cut_pic_widget = QWidget()
        cut_pic_widget.setLayout(cut_pic_layout)
        self.result_label = QLabel()
        self.result_label.setText('识别结果:')
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(80)
        self.result_text.setMaximumWidth(240)

        d_pic_label = QLabel()
        d_pic_label.setText('切割结果:')

        d_pic_layout = QHBoxLayout()
        self.d_pic1 = QLabel()
        d_pic_layout.addWidget(self.d_pic1)
        self.d_pic2 = QLabel()
        d_pic_layout.addWidget(self.d_pic2)
        self.d_pic3 = QLabel()
        d_pic_layout.addWidget(self.d_pic3)
        self.d_pic4 = QLabel()
        d_pic_layout.addWidget(self.d_pic4)
        self.d_pic5 = QLabel()
        d_pic_layout.addWidget(self.d_pic5)
        self.d_pic6 = QLabel()
        d_pic_layout.addWidget(self.d_pic6)
        self.d_pic7 = QLabel()
        d_pic_layout.addWidget(self.d_pic7)
        d_pic_widget = QWidget()
        d_pic_widget.setLayout(d_pic_layout)

        result_layout = QVBoxLayout()
        result_layout.addWidget(self.result_label)
        result_layout.addWidget(self.result_text)
        result_widget = QWidget()
        result_widget.setLayout(result_layout)

        right_layout.addWidget(cut_pic_widget)
        right_layout.addWidget(result_widget)
        right_layout.addWidget(d_pic_label)
        right_layout.addWidget(d_pic_widget)
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
        reco_button.setText("单张识别_unet")
        reco_button.clicked.connect(lambda: self.open_img())
        reco_button2 = QPushButton()
        reco_button2.setText("单张识别_yolo")
        reco_button2.clicked.connect(lambda: self.open_img_yolo())
        reco_all_button = QPushButton()
        reco_all_button.setText("集中识别")
        reco_all_button.clicked.connect(lambda: self.pred_all())
        button_layout1.addWidget(reco_button)
        button_layout1.addWidget(reco_button2)
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

        button_widget3 = QWidget()
        button_layout3 = QHBoxLayout()
        dz_button = QPushButton()
        dz_button.setText("单张识别_切割")
        dz_button.clicked.connect(lambda: self.danzhangqiege())
        dzq_button = QPushButton()
        dzq_button.setText("集中识别_切割")
        dzq_button.clicked.connect(lambda: self.jizhongqiege())
        button_layout3.addWidget(dz_button)
        button_layout3.addWidget(dzq_button)
        button_widget3.setLayout(button_layout3)

        button_layout.addWidget(button_widget1)
        button_layout.addWidget(button_widget2)
        button_layout.addWidget(button_widget3)

        button_widget.setLayout(button_layout)

        return button_widget

    def predict_img(self, img_cut, img_label):
        height, width, _ = img_cut.shape
        bytesPerLine = 3 * width
        qImg = QImage(img_cut.data.tobytes(), width, height, bytesPerLine,
                      QImage.Format_BGR888)
        self.cut_pic.setPixmap(QPixmap(qImg).scaled(240, 80))

        height, width, _ = img_label.shape
        bytesPerLine = 3 * width
        qImg2 = QImage(img_label.data.tobytes(), width, height, bytesPerLine,
                       QImage.Format_BGR888)
        self.ori_pic.setPixmap(QPixmap(qImg2).scaled(373, 600))

        result = cnn_predict(self.cnn, img_cut)
        self.result = result[1]

    def predict_img2(self, img_cut, img_label):
        height, width, _ = img_cut.shape
        bytesPerLine = 3 * width
        qImg = QImage(img_cut.data.tobytes(), width, height, bytesPerLine,
                      QImage.Format_BGR888)

        height, width, _ = img_label.shape
        bytesPerLine = 3 * width
        qImg2 = QImage(img_label.data.tobytes(), width, height, bytesPerLine,
                       QImage.Format_BGR888)
        self.cut_pic.setPixmap(
            QPixmap(qImg).scaled(240, 80, Qt.KeepAspectRatio))
        self.ori_pic.setPixmap(QPixmap(qImg2).scaled(373, 600))

        try:
            qiegechepai = qgHS.qg(img_cut)
        except:
            msg = QMessageBox(QMessageBox.Critical, '错误', '切割错误')
            msg.exec_()

        if len(qiegechepai) == 7:
            height, width = qiegechepai[0].shape
            qImg3 = QImage(qiegechepai[0].data.tobytes(), width, height, width,
                           QImage.Format_Grayscale8)
            self.d_pic1.setPixmap(QPixmap(qImg3).scaled(32, 48))
            Qimg3 = qiegechepai[0]
            a = ZF.danGeZiFuShiBie(Qimg3)

            height, width = qiegechepai[1].shape
            qImg4 = QImage(qiegechepai[1].data.tobytes(), width, height, width,
                           QImage.Format_Grayscale8)
            self.d_pic2.setPixmap(
                QPixmap(qImg4).scaled(32, 48, Qt.KeepAspectRatio))
            Qimg4 = qiegechepai[1]
            b = ZF.danGeZiFuShiBie(Qimg4)

            height, width = qiegechepai[2].shape
            qImg5 = QImage(qiegechepai[2].data.tobytes(), width, height, width,
                           QImage.Format_Grayscale8)
            self.d_pic3.setPixmap(
                QPixmap(qImg5).scaled(32, 48, Qt.KeepAspectRatio))
            Qimg5 = qiegechepai[2]
            c = ZF.danGeZiFuShiBie(Qimg5)

            height, width = qiegechepai[3].shape
            qImg6 = QImage(qiegechepai[3].data.tobytes(), width, height, width,
                           QImage.Format_Grayscale8)
            self.d_pic4.setPixmap(
                QPixmap(qImg6).scaled(32, 48, Qt.KeepAspectRatio))
            Qimg6 = qiegechepai[3]
            d = ZF.danGeZiFuShiBie(Qimg6)

            height, width = qiegechepai[4].shape
            qImg7 = QImage(qiegechepai[4].data.tobytes(), width, height, width,
                           QImage.Format_Grayscale8)
            self.d_pic5.setPixmap(
                QPixmap(qImg7).scaled(32, 48, Qt.KeepAspectRatio))
            Qimg7 = qiegechepai[4]
            e = ZF.danGeZiFuShiBie(Qimg7)

            height, width = qiegechepai[5].shape
            qImg8 = QImage(qiegechepai[5].data.tobytes(), width, height, width,
                           QImage.Format_Grayscale8)
            self.d_pic6.setPixmap(
                QPixmap(qImg8).scaled(32, 48, Qt.KeepAspectRatio))
            Qimg8 = qiegechepai[5]
            f = ZF.danGeZiFuShiBie(Qimg8)

            height, width = qiegechepai[6].shape
            qImg9 = QImage(qiegechepai[6].data.tobytes(), width, height, width,
                           QImage.Format_Grayscale8)
            self.d_pic7.setPixmap(
                QPixmap(qImg9).scaled(32, 48, Qt.KeepAspectRatio))
            Qimg9 = qiegechepai[6]
            g = ZF.danGeZiFuShiBie(Qimg9)

            self.result = str(a) + str(b) + str(c) + str(d) + str(e) + str(
                f) + str(g)
            saveimg = cv2.imread(self.path)
            #savepath = 'D:\computer_vision\CarPlateRecog-All-In-One\\basetupian\\' +
            #cv2.imwrite(saveimg,r'D:\computer_vision\CarPlateRecog-All-In-One\basetupian')
        else:
            print("切割错误")
            print(len(qiegechepai))
            self.result = 'qecw'

        # result = cnn_predict(self.cnn, img_cut)
        # self.result = "result[1]"

    def locate_img(self):
        start = time.time()
        img = cv2.imread(self.path)
        try:
            if self.way == 'unet':
                img_cut, img_label = locate(img, self.unet)
                self.predict_img(img_cut, img_label)
            elif self.way == 'shoudong':
                img_cut, img_label = detect(img, self.yolo, self.device)
                self.predict_img2(img_cut, img_label)
            else:
                img_cut, img_label = detect(img, self.yolo, self.device)
                self.predict_img(img_cut, img_label)
            end = time.time()
            self.text += self.result + '\t  ' + str('%.2f' %
                                                    (end - start)) + 's\n'
            self.dialog.setText(self.text)
            self.result_text.setText(self.result + '\n' + str(end - start) +
                                     's')
        except Exception as e:
            msg = QMessageBox(QMessageBox.Critical, '错误', '定位失败')
            msg.exec_()
            height, width, _ = img.shape
            bytesPerLine = 3 * width
            qImg = QImage(img.data.tobytes(), width, height, bytesPerLine,
                          QImage.Format_BGR888)
            self.ori_pic.setPixmap(QPixmap(qImg).scaled(373, 600))

    def open_img(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "open file", "", "ALL FILES (*.*);;PHOTO FILES (*.jpg)")

        if path:
            self.path = path
            self.way = 'unet'
            self.locate_img()

    def translate(self, label):
        provinces = [
            "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京",
            "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏",
            "陕", "甘", "青", "宁", "新", "警", "学", "O"
        ]
        ads = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N',
            'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1',
            '2', '3', '4', '5', '6', '7', '8', '9', 'O'
        ]

        plate = ""
        for i, number in enumerate(label.split("_")):
            if i == 0:
                temp = provinces[int(number)]
            else:
                temp = ads[int(number)]
            plate += str(temp)
        return plate

    def pred_all(self):
        self.text = ''
        path = QFileDialog.getExistingDirectory(self, "choose folder", "./")

        if path:
            items = ["unet", "yolo"]
            value, ok = QInputDialog.getItem(self, "choose way", "请选择定位方式:",
                                             items, 1, True)
            if ok:
                start = time.time()
                self.way = value
                picName = os.listdir(path)
                num_of_test = 0
                count = 0
                for i in picName:
                    if i.split('.')[1] == 'jpg':
                        num_of_test += 1
                        self.path = path + '/' + i
                        label = i.rsplit('.', 1)[0].split('-')[-3]
                        label = self.translate(label)
                        self.locate_img()
                        if self.result == label:
                            count += 1
                            print(label + ' √')
                            self.result_text.setText(
                                self.result_text.toPlainText() + '\n√')
                            self.text = self.text[:-1]
                            self.text += ' √\n'
                            self.dialog.setText(self.text)
                        else:
                            print(self.result + ' ×')
                            self.result_text.setText(
                                self.result_text.toPlainText() + '\n×')
                            self.text = self.text[:-1]
                            self.text += ' ×\n'
                            self.dialog.setText(self.text)
                        if num_of_test != 1:
                            QApplication.processEvents()
                end = time.time()
                result = count / num_of_test
                self.result_text.setText(
                    str(count) + '/' + str(num_of_test) + ', ' + str(result) +
                    '\n' + str(end - start) + 's')

    def jizhongqiege(self):
        self.text = ''
        path = QFileDialog.getExistingDirectory(self, "choose folder", "./")

        writepath = 'D:\computer_vision\CarPlateRecog-All-In-One\\basetupian'

        if path:
            start = time.time()
            self.way = 'shoudong'
            picName = os.listdir(path)
            num_of_test = 0
            count = 0
            for i in picName:
                if i.split('.')[1] == 'jpg':
                    self.path = path + '/' + i
                    label = i.rsplit('.', 1)[0].split('-')[-3]
                    label = self.translate(label)
                    self.locate_img()
                    relist = list(self.result)
                    lalist = list(label)
                    if len(relist) == 7:
                        # writepath2 = writepath + '\\' + i
                        # writeimg = cv2.imread(self.path)
                        # cv2.imwrite(writepath2,writeimg)
                        for i in range(7):
                            num_of_test += 1
                            if relist[i] == lalist[i]:
                                count += 1
                    else:
                        continue

                    QApplication.processEvents()
            end = time.time()
            result = count / num_of_test
            self.result_text.setText(
                str(count) + '/' + str(num_of_test) + ', ' + str(result) +
                '\n' + str(end - start) + 's')

    def open_img_yolo(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "open file", "", "ALL FILES (*.*);;PHOTO FILES (*.jpg)")

        if path:
            self.path = path
            self.way = 'yolo'
            self.locate_img()

    def pic_up(self):
        try:
            path = os.path.split(self.path)
            picName = os.listdir(path[0])
            n = picName.index(path[1])
            n = n - 1
            if n < 0:
                n = len(picName) - 1
            self.path = path[0] + '/' + picName[n]
            if picName[n].split('.')[1] == 'jpg':
                self.locate_img()
            else:
                self.pic_up()
        except:
            msg = QMessageBox(QMessageBox.Critical, '错误', '未读入图片')
            msg.exec_()

    def pic_down(self):
        try:
            path = os.path.split(self.path)
            picName = os.listdir(path[0])
            n = picName.index(path[1])
            n = n + 1
            if n >= len(picName):
                n = 0
            self.path = path[0] + '/' + picName[n]
            if picName[n].split('.')[1] == 'jpg':
                self.locate_img()
            else:
                self.pic_down()
        except:
            msg = QMessageBox(QMessageBox.Critical, '错误', '未读入图片')
            msg.exec_()

    def danzhangqiege(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "open file", "", "ALL FILES (*.*);;PHOTO FILES (*.jpg)")

        if path:
            self.path = path
            self.way = 'shoudong'
            self.locate_img()


if __name__ == "__main__":
    app = QApplication([])
    window = car_recog_gui()
    window.show()
    sys.exit(app.exec())
