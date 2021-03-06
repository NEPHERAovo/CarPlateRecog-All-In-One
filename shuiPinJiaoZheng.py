'''
created on 06/24/2022
author: 钟余盛
description: 车牌分割矫正函数
'''

# coding=utf-8
import cv2
import numpy as np
import math
from PIL import Image


# 度数转换
def DegreeTrans(theta):
    res = theta / np.pi * 180
    return res


# 逆时针旋转图像degree角度（原尺寸）
def rotateImage(src, degree):
    # 旋转中心为图像中心
    h, w = src.shape[:2]
    # 计算二维旋转的仿射变换矩阵
    RotateMatrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), degree, 1)
    nH = int(abs(h * math.cos(math.radians(degree))) + abs(w * math.sin(math.radians(degree))))
    nW = int(abs(h * math.sin(math.radians(degree))) + abs(w * math.cos(math.radians(degree))))

    #print(RotateMatrix)
    # 仿射变换，背景色填充为黑色
    rotate = cv2.warpAffine(src, RotateMatrix, (nW, nH), borderValue=(0, 0))
    return rotate


# 通过霍夫变换计算角度
def CalcDegree(srcImage):
    #midImage = cv2.cvtColor(srcImage, cv2.COLOR_BGR2GRAY)
    dstImage = cv2.Canny(srcImage, 50, 200, 3)
    lineimage = srcImage.copy()

    #cv2.imshow('a0', dstImage)

    # 通过霍夫变换检测直线
    # 第4个参数就是阈值，阈值越大，检测精度越高,经过测试，使用70为值域可以调整大多数车牌的倾斜角
    lines = cv2.HoughLines(dstImage, 1, np.pi / 180, 70)
    # 由于图像不同，阈值不好设定，因为阈值设定过高导致无法检测直线，阈值过低直线太多，速度很慢
    sum = 0
    # 依次画出每条线段
    if lines is not None:
        for i in range(len(lines)):
            for rho, theta in lines[i]:
                # print("theta:", theta, " rho:", rho)
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(round(x0 + 1000 * (-b)))
                y1 = int(round(y0 + 1000 * a))
                x2 = int(round(x0 - 1000 * (-b)))
                y2 = int(round(y0 - 1000 * a))
                # 只选角度最小的作为旋转角度
                sum += theta
                cv2.line(lineimage, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
            #cv2.imshow("Imagelines", lineimage)

        # 对所有角度求平均，这样做旋转效果会更好
        average = sum / len(lines)
        angle = DegreeTrans(average) - 90
        return angle
    else:
        return 0


'''
if __name__ == '__main__':
    image = cv2.imread(r'D:\computer_vision\CarPlateRecog-All-In-One\aaT\8.jpg')
    cv2.imshow("Image", image)
    # 倾斜角度矫正
    degree = CalcDegree(image)
    print("调整角度：", degree)
    rotate = rotateImage(image, degree)
    cv2.imshow("rotate", rotate)
    # cv2.imwrite("../test/recified.png", rotate, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''