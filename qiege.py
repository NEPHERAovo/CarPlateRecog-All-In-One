import os
import imghdr
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure,color
import guangZhaoXiuGai as gzxg
import imutils
import shuiPinJiaoZheng as sp
import hengXiangQieGe as hx

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False


dir = r'D:\computer_vision\CarPlateRecog-All-In-One\chepaitupian'
imgList = os.listdir(dir)
print(imgList)
imgList.sort(key=lambda x: (x.replace("frame","").split('.')[0]))
print(imgList)
wpath = r'D:\computer_vision\CarPlateRecog-All-In-One\aaT'
for count in range(0, len(imgList)):
    im_name = imgList[count]
    im_path = os.path.join(dir,im_name)
    print(im_path)
    img = cv2.imdecode(np.fromfile(im_path, dtype=np.uint8), -1)
    #img = cv2.imread(im_path)
    gray = gzxg.unevenLightCompensate(img,16)##调用改善图片光照的函数进行图片的灰度化


    #使用最佳全局阈值比较适应大多数车牌
    ret, ostu_thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(os.path.join(wpath, str(count) + '.jpg'), ostu_thresh_img)
    plt.figure()
    plt.subplot(221), plt.imshow(ostu_thresh_img, cmap='gray'), plt.title("二值")
    plt.subplot(222), plt.imshow(gray, cmap='gray'), plt.title("灰度")
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(223), plt.imshow(image, cmap='gray'), plt.title("原图")

    #霍夫变换计算角度矫正
    degree = sp.CalcDegree(ostu_thresh_img)
    rotate_img = sp.rotateImage(ostu_thresh_img, degree)
    # ret1, rotate_img1 = cv2.threshold(rotate_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plt.subplot(224), plt.imshow(rotate_img, cmap='gray'), plt.title("水平矫正图")
    ret, rotate_img2 = cv2.threshold(rotate_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)



    #横向切割
    hqieg_img = hx.remove_upanddown_border(rotate_img2)
    plt.figure()
    plt.subplot(221), plt.imshow(hqieg_img, cmap='gray'), plt.title("横向切割")
    cv2.imwrite(os.path.join(wpath, str(count)+ 'a' + '.jpg'), hqieg_img)

    #纵向切割
    shuqie = hx.get_wave_peaks(hqieg_img)
    dange = hx.seperate_card(hqieg_img,shuqie)
    plt.figure()
    for x in range(len(dange)):
        plt.subplot(1,len(dange),x+1), plt.imshow(dange[x], cmap='gray'), plt.title(x)





    plt.show()
    cv2.waitKey()
    cv2.destroyAllWindows()