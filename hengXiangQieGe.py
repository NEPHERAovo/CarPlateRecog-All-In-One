import cv2
import numpy as np

def find_waves(threshold, histogram):
    """ 根据设定的阈值和图片直方图，找出波峰，用于分隔字符 """
    up_point = -1  # 上升点
    is_peak = False
    if histogram[0] > threshold:
        up_point = 0
        is_peak = True
    wave_peaks = []
    for i, x in enumerate(histogram):
        if is_peak and x < threshold:
            if i - up_point > 2:
                is_peak = False
                wave_peaks.append((up_point, i))
        elif not is_peak and x >= threshold:
            is_peak = True
            up_point = i
    if is_peak and up_point != -1 and i - up_point > 4:
        wave_peaks.append((up_point, i))
    return wave_peaks


def remove_upanddown_border(img):
    """ 去除车牌上下无用的边缘部分，确定上下边界 """
    plate_binary_img = img
    row_histogram = np.sum(plate_binary_img, axis=1)  # 数组的每一行求和
    row_min = np.min(row_histogram)
    row_average = np.sum(row_histogram) / plate_binary_img.shape[0]

    row_average1 = row_average/255.0
    if (row_average1>100):
        row_threshold = row_average * 0.75
    elif (row_average1<=100 and row_average1>90):
        row_threshold = row_average * 0.725
    elif (row_average1<=90 and row_average1>80):
        row_threshold = row_average * 0.7
    elif (row_average1<=80 and row_average1>70):
        row_threshold = row_average * 0.65
    elif (row_average1<=70 and row_average1>60):
        row_threshold = row_average * 0.6
    else:
        row_threshold = row_average * 0.35

    #row_threshold = (row_min + row_average) / 2
    wave_peaks = find_waves(row_threshold, row_histogram)
    # 挑选跨度最大的波峰
    wave_span = 0.0
    for wave_peak in wave_peaks:
        span = wave_peak[1] - wave_peak[0]
        if span > wave_span:
            wave_span = span
            selected_wave = wave_peak
    plate_binary_img = plate_binary_img[selected_wave[0]:selected_wave[1], :]
    # cv.imshow("plate_binary_img", plate_binary_img)
    return plate_binary_img


#竖直分割

# 获取波峰
def get_wave_peaks(gray_img):
    y_histogram = np.sum(gray_img, axis=0)
    y_min = np.min(y_histogram)
    y_average = np.sum(y_histogram) / y_histogram.shape[0]
    y_threshold = (y_min + y_average) / 4.5  # U和0要求阈值偏小，否则U和0会被分成两半
    wave_peaks = find_waves(y_threshold, y_histogram)

    wave = max(wave_peaks, key=lambda x: x[1] - x[0])
    max_wave_dis = wave[1] - wave[0]
    # 判断是否是左侧车牌边缘
    if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis / 4 and wave_peaks[0][0] == 0:
        wave_peaks.pop(0)

    # 组合分离汉字
    cur_dis = 0
    for i, wave in enumerate(wave_peaks):
        if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.6 or len(wave_peaks) <= 6:
            break
        else:
            cur_dis += wave[1] - wave[0]
    if i > 0:
        wave = (wave_peaks[0][0], wave_peaks[i][1])
        wave_peaks = wave_peaks[i + 1:]
        wave_peaks.insert(0, wave)
    # 去除车牌上的分隔点
    if len(wave_peaks) > 2:
        point = wave_peaks[2]  #认为第三个点为点
        if point[1] - point[0] < max_wave_dis / 3:
            point_img = gray_img[:, point[0]:point[1]]
            if np.mean(point_img) < 255 / 5:
                wave_peaks.pop(2)

    for j in range(len(wave_peaks)):
        if j <= len(wave_peaks)-2:
            #if wave_peaks[j][1] - wave_peaks[j][0] < max_wave_dis / 2 and (wave_peaks[j][0]-wave_peaks[j-1][1] < 4 or wave_peaks[j+1][0]-wave_peaks[j][1] < 4):
            point = wave_peaks[j]
            point_img = gray_img[:, point[0]:point[1]]
            if wave_peaks[j][1] - wave_peaks[j][0] < max_wave_dis / 1.75:
                if np.mean(point_img) < 255 / 2.9:  #排除是一的可能
                    wave_peaks.pop(j)

            #if wave_peaks[j][0]-wave_peaks[j-1][1] < 4:
#连接字符
            if j <= len(wave_peaks) - 2:
                if wave_peaks[j+1][0]-wave_peaks[j][1] < 4:
                    h_dis = wave_peaks[j+1][1]-wave_peaks[j][0]
                    if h_dis < max_wave_dis / 0.9:
                        wave = (wave_peaks[j][0], wave_peaks[j+1][1])
                        wave_peaks.pop(j)
                        wave_peaks.insert(j, wave)
                        wave_peaks.pop(j+1)


    wave_peaks = wave_peaks[0:7]


    return wave_peaks

def seperate_card(img, waves):
    part_cards = []      #分割出的图像列表
    for wave in waves:
        part_cards.append(img[:, wave[0]:wave[1]])  #按照波峰提取图像添加入列表
    return part_cards
