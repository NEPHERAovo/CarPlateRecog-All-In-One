'''
created on 06/24/2022
author: 钟余盛
description: 车牌预处理光照修改函数
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

def unevenLightCompensate(img, blockSize):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gray = cv2.equalizeHist(gray)##直方图均衡
    #gray = cv2.medianBlur(gray, 3)##中值降噪
    average = np.mean(gray)

    rows_new = int(np.ceil(gray.shape[0] / blockSize))
    cols_new = int(np.ceil(gray.shape[1] / blockSize))

    blockImage = np.zeros((rows_new, cols_new), dtype=np.float32)
    for r in range(rows_new):
        for c in range(cols_new):
            rowmin = r * blockSize
            rowmax = (r + 1) * blockSize
            if (rowmax > gray.shape[0]):
                rowmax = gray.shape[0]
            colmin = c * blockSize
            colmax = (c + 1) * blockSize
            if (colmax > gray.shape[1]):
                colmax = gray.shape[1]

            imageROI = gray[rowmin:rowmax, colmin:colmax]
            temaver = np.mean(imageROI)
            blockImage[r, c] = temaver

    blockImage = blockImage - average
    blockImage2 = cv2.resize(blockImage, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
    gray2 = gray.astype(np.float32)
    dst = gray2 - blockImage2
    dst = dst.astype(np.uint8)
    dst = cv2.medianBlur(dst, 3)
    #dst = cv2.GaussianBlur(dst, (3, 3), 0)
    #dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    return dst



'''
if __name__ == '__main__':
    file = r'D:\computer_vision\CarPlateRecog-All-In-One\aTestForOnePecture\1.jpg'
    blockSize = 16
    img = cv2.imread(file)
    dst = unevenLightCompensate(img, blockSize)

    #result = np.concatenate([img, dst], axis=1)
    plt.figure()
    plt.subplot(221), plt.imshow(dst, cmap='gray'), plt.title("-")
    plt.show()
    #cv2.imshow('result', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''






