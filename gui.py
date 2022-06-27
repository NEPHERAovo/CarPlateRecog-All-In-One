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
        result_layout = QVBoxLayout()
        result_layout.addWidget(self.result_label)
        result_layout.addWidget(self.result_text)
        result_widget = QWidget()
        result_widget.setLayout(result_layout)

        right_layout.addWidget(cut_pic_widget)
        right_layout.addWidget(result_widget)
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

        button_layout.addWidget(button_widget1)
        button_layout.addWidget(button_widget2)

        button_widget.setLayout(button_layout)

        return button_widget

    def predict_img(self, img_cut, img_label):
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
        result = cnn_predict(self.cnn, img_cut)
        self.result = result[1]

    def locate_img(self):
        start = time.time()
        img = cv2.imread(self.path)
        if self.way == 'unet':
            img_cut, img_label = locate(img, self.unet)
            self.predict_img(img_cut, img_label)
        else:
            img_cut, img_label = detect(img, self.yolo, self.device)
            self.predict_img(img_cut, img_label)
        end = time.time()
        self.text += self.result + '\t  ' + str('%.2f' % (end - start)) + 's\n'
        self.dialog.setText(self.text)
        self.result_text.setText(self.result + '\n' + str(end - start) + 's')

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
                        else:
                            print(self.result + ' ×')
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


if __name__ == "__main__":
    app = QApplication([])
    window = car_recog_gui()
    window.show()
    sys.exit(app.exec())
