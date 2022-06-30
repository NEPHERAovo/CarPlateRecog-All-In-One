'''
created on 06/20/2022
author: 杨宇轩
description: cnn识别/训练/测试
'''

import os
import cv2
import keras
import numpy as np
from keras import layers, models

# 字符字典，读入时用
characters_dict = {
    "京": 0,
    "沪": 1,
    "津": 2,
    "渝": 3,
    "冀": 4,
    "晋": 5,
    "蒙": 6,
    "辽": 7,
    "吉": 8,
    "黑": 9,
    "苏": 10,
    "浙": 11,
    "皖": 12,
    "闽": 13,
    "赣": 14,
    "鲁": 15,
    "豫": 16,
    "鄂": 17,
    "湘": 18,
    "粤": 19,
    "桂": 20,
    "琼": 21,
    "川": 22,
    "贵": 23,
    "云": 24,
    "藏": 25,
    "陕": 26,
    "甘": 27,
    "青": 28,
    "宁": 29,
    "新": 30,
    "0": 31,
    "1": 32,
    "2": 33,
    "3": 34,
    "4": 35,
    "5": 36,
    "6": 37,
    "7": 38,
    "8": 39,
    "9": 40,
    "A": 41,
    "B": 42,
    "C": 43,
    "D": 44,
    "E": 45,
    "F": 46,
    "G": 47,
    "H": 48,
    "J": 49,
    "K": 50,
    "L": 51,
    "M": 52,
    "N": 53,
    "P": 54,
    "Q": 55,
    "R": 56,
    "S": 57,
    "T": 58,
    "U": 59,
    "V": 60,
    "W": 61,
    "X": 62,
    "Y": 63,
    "Z": 64
}
# 字符，预测时读取正确车牌用
characters = [
    "京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣",
    "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
    "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D",
    "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U",
    "V", "W", "X", "Y", "Z"
]


# 测试神经网络
def cnn_test(PATH_TEST_DIR='D:\Softwares\Python\croppedTEST/'):
    # 初始化计数
    acc = 0
    fal = 0
    withoutChn = 0
    # 载入神经网络
    cnn = keras.models.load_model('cnn.h5')
    picName = os.listdir(PATH_TEST_DIR)
    n = len(picName)
    # 遍历路径下所有文件
    for i in range(n):
        # 读取图片
        pic = cv2.imdecode(
            np.fromfile(PATH_TEST_DIR + picName[i], dtype=np.uint8), -1)
        # 读取字符
        label = [characters_dict[char] for char in picName[i][0:7]]
        # 读入正确结果
        predict = ''
        for j in range(0, 7):
            predict = predict + characters[label[j]]
        # 得到预测结果
        res = cnn_predict(cnn, pic)
        print('predict:' + res[1] + '     answer:' + predict, end='')
        # 如果正确
        if res[1] == predict:
            acc = acc + 1
            print('   √')
        else:
            fal = fal + 1
            print('   ×')
        if res[1][2:] == predict[2:]:
            withoutChn = withoutChn + 1
    # 显示测试结果
    print("acc = %d , fal = %d, acc without province = %d" %
          (acc, fal, withoutChn))
    print("accuracy = %lf" % (acc / i))
    print("accuracy without province = %lf" % (withoutChn / i))


def cnn_train(PATH_TRAIN_DIR='D:\Softwares\Python\croppedTEST/'):

    # 读入训练集
    picName = os.listdir(PATH_TRAIN_DIR)
    n = len(picName)
    # 初始化训练集
    x_train, y_train = [], []
    for i in range(n):
        if i == 0:
            print("开始读取图片...")
        elif i % 5000 == 0:
            print("已读取%d张图片" % i)
        # 文件名包含中文，使用imdecode的方式以uint8打开，flags = -1表示打开源文件
        pic = cv2.imdecode(
            np.fromfile(PATH_TRAIN_DIR + picName[i], dtype=np.uint8), -1)
        # label存入字符在characters_dict中对应的数字
        label = [characters_dict[char] for char in picName[i][0:7]]
        x_train.append(pic)
        y_train.append(label)
    print("已读取%d张图片" % i)

    # 将训练集转为numpy标准数组格式
    x_train = np.array(x_train)
    # y_train由 n x 7矩阵 变为 7 x n矩阵
    y_train = [np.array(y_train)[:, i] for i in range(7)]

    # 输入层
    inputLayer = layers.Input((80, 240, 3))
    x = inputLayer

    # 卷积 归一化 池化
    x = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=2, strides=2)(x)

    x = layers.Conv2D(filters=64, kernel_size=3, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=2, strides=2)(x)

    x = layers.Conv2D(filters=128, kernel_size=3, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=2, strides=2)(x)

    x = layers.Conv2D(filters=192, kernel_size=3, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=2, strides=2)(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)

    # 7个全连接层，每个对应65个字符， 总loss = loss1 + ... + loss7
    outputLayer = [
        layers.Dense(65, activation='softmax', name='c%d' % (i + 1))(x)
        for i in range(7)
    ]

    model = models.Model(inputs=inputLayer, outputs=outputLayer)
    # 输出各层参数
    model.summary()
    # 配置训练模型
    model.compile(
        optimizer='adam',
        # 多分类交叉熵损失函数，数字类label
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    # 训练
    print("start training")
    model.fit(x_train, y_train, epochs=10)
    model.save('cnn.h5')
    print('saved file to cnn.h5')


# 预测
def cnn_predict(cnn, img):
    predict = []
    lic = img

    # crop为指定大小
    temp = cnn.predict(lic.reshape(1, 80, 240, 3))
    temp = np.array(temp).reshape(7, 65)

    chars = ''
    # 取最大可能性
    for arg in np.argmax(temp, axis=1):
        chars += characters[arg]

    # 车牌原图放入
    predict.append(lic)
    # 预测结果放入
    predict.append(chars)
    return predict
