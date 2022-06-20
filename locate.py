from keras import models
import cv2
import numpy as np
from skimage.measure import label
from cnn_reco import cnn_predict

import os
from keras import layers, losses, models
import numpy as np
import cv2


def cnn_train():
    # 读取数据集
    path = 'D:/Softwares/Python/CV/'
    pic_name = sorted(os.listdir(path + '/1'))
    n = len(pic_name)
    X_train, y_train = [], []
    for i in range(n):
        if i % 200 == 0:
            print("已读取%d张图片" % i)
        img = cv2.imread(path + '/1/' + pic_name[i])
        label = cv2.imread(path + '/2/' + pic_name[i])
        X_train.append(img)
        y_train.append(label)
    print("已读取%d张图片" % i)
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same'):
        x = layers.Conv2D(nb_filter, kernel_size,
                          strides=strides, padding=padding)(x)
        x = layers.BatchNormalization(axis=3)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        return x

    def Conv2dT_BN(x, filters, kernel_size, strides=(2, 2), padding='same'):
        x = layers.Conv2DTranspose(
            filters, kernel_size, strides=strides, padding=padding)(x)
        x = layers.BatchNormalization(axis=3)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        return x

    inpt = layers.Input(shape=(512, 512, 3))

    conv1 = Conv2d_BN(inpt, 16, (3, 3))
    conv1 = Conv2d_BN(conv1, 16, (3, 3))
    pool1 = layers.MaxPooling2D(pool_size=(
        2, 2), strides=(2, 2), padding='same')(conv1)

    conv2 = Conv2d_BN(pool1, 32, (3, 3))
    conv2 = Conv2d_BN(conv2, 32, (3, 3))
    pool2 = layers.MaxPooling2D(pool_size=(
        2, 2), strides=(2, 2), padding='same')(conv2)

    conv3 = Conv2d_BN(pool2, 64, (3, 3))
    conv3 = Conv2d_BN(conv3, 64, (3, 3))
    pool3 = layers.MaxPooling2D(pool_size=(
        2, 2), strides=(2, 2), padding='same')(conv3)

    convt1 = Conv2dT_BN(pool3, 64, (3, 3))
    concat1 = layers.concatenate([conv3, convt1], axis=3)
    concat1 = layers.Dropout(0.5)(concat1)
    conv4 = Conv2d_BN(concat1, 64, (3, 3))
    conv4 = Conv2d_BN(conv4, 64, (3, 3))

    convt2 = Conv2dT_BN(conv4, 32, (3, 3))
    concat2 = layers.concatenate([conv2, convt2], axis=3)
    concat2 = layers.Dropout(0.5)(concat2)
    conv5 = Conv2d_BN(concat2, 32, (3, 3))
    conv5 = Conv2d_BN(conv5, 32, (3, 3))

    convt3 = Conv2dT_BN(conv5, 16, (3, 3))
    concat3 = layers.concatenate([conv1, convt3], axis=3)
    concat3 = layers.Dropout(0.5)(concat3)
    conv6 = Conv2d_BN(concat3, 16, (3, 3))
    conv6 = Conv2d_BN(conv6, 16, (3, 3))

    conv7 = layers.Dropout(0.5)(conv6)
    outpt = layers.Conv2D(filters=3, kernel_size=(1, 1), strides=(
        1, 1), padding='same', activation='relu')(conv7)

    model = models.Model(inpt, outpt)
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    # 模型训练
    print("开始训练")
    model.fit(X_train, y_train, epochs=10, batch_size=5)
    model.save('locate.h5')
    print('locate.h5保存成功')


def locate(img, unet):
    img = cv2.resize(img, (512, 512))
    img_copy = img.copy()
    img = img.reshape(1, 512, 512, 3)
    predict = unet.predict(img)
    predict = predict.reshape(512, 512, 3)
    _, binary = cv2.threshold(predict, 127, 255, cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8)

    contours, _ = cv2.findContours(
        binary[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 1:
        labeled_img, num = label(binary, background=0, return_num=True)
        max_label = 0
        max_num = 0
        for i in range(1, num+1):
            if np.sum(labeled_img == i) > max_num:
                max_num = np.sum(labeled_img == i)
                max_label = i
        final_img = (labeled_img == max_label).astype(np.uint8)
        contours, _ = cv2.findContours(
            final_img[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    img_cut = img_copy[y:y+h, x:x+w]
    img_cut = cv2.resize(img_cut, (240, 80))

    cv2.drawContours(img_copy, [contours[0]], 0, (0, 0, 255), 2)
    # return img_cut, img_copy, predict
    return img_cut, img_copy


if __name__ == '__main__':
    # cnn = models.load_model('cnn.h5')
    unet = models.load_model('locate.h5')
    img = cv2.imread('xxx.jpg')
    img_cut, img = locate(img, unet)
    cv2.imshow('ori', img)
    cv2.imshow('cut plate', img_cut)
    # cv2.imshow('predict bin', predict)
    cv2.waitKey()
    cv2.destroyAllWindows()
    # print(cnn_predict(cnn, img_cut))

# cv2.imshow('1', img_cut)
# cv2.waitKey()
# cv2.destroyAllWindows()


# 左上角，左下角，右上角，右下角，形成的新box顺序需和原box中的顺序对应，以进行转换矩阵的形成
    # p0 = np.float32([l0, l1, l2, l3])
    # p1 = np.float32([(0, 0), (0, 80), (240, 0), (240, 80)])
    # transform_mat = cv2.getPerspectiveTransform(p0, p1)  # 构成转换矩阵
    # lic = cv2.warpPerspective(img_src, transform_mat, (240, 80))  # 进行车牌矫正
# cv2.polylines(img_copy,cv2.boxPoints(rect))
    # rect = cv2.minAreaRect(contours[0])
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
