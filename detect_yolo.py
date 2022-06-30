import os
import sys
from pathlib import Path
import numpy as np
import cv2
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
sys.path.append(str(ROOT) + '\\yolov5')  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov5.utils.augmentations import letterbox
from yolov5.utils.torch_utils import select_device
from yolov5.utils.general import (check_img_size, cv2, non_max_suppression,
                                  scale_coords, xyxy2xywh)
from yolov5.models.common import DetectMultiBackend


def detect(img, model, device):
    imgsz = (512, 512)

    stride, names = model.stride, model.names

    imgsz = check_img_size(imgsz, s=stride)

    model.model.float()
    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz))  # warmup

    im0 = img
    # Padded resize
    im = letterbox(im0, new_shape=imgsz)[0]

    # Convert
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)

    im = torch.from_numpy(im).to(device)
    im = im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    # Inference
    pred = model(im)
    # NMS
    pred = non_max_suppression(pred)

    # 用于存放结果
    detections = []
    # Process predictions
    for i, det in enumerate(pred):  # per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4],
                                      im0.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(
                    1, 4))).view(-1).tolist()
                xywh = [round(x) for x in xywh]
                xywh = [
                    xywh[0] - xywh[2] // 2, xywh[1] - xywh[3] // 2, xywh[2],
                    xywh[3]
                ]  # 检测到目标位置（x，y，w，h）
                cls = names[int(cls)]
                conf = float(conf)
                detections.append({
                    'class': cls,
                    'conf': conf,
                    'position': xywh
                })
    if len(detections) == 0:
        path = "D:\computer_vision\CarPlateRecog-All-In-One\\aaT\\wwww.jpg"
        cv2.imwrite(path, img)
        #raise Exception('dinWeiChuoWu')
    img_cut = img[xywh[1]:xywh[1] + xywh[3], xywh[0]:xywh[0] + xywh[2]]
    img_cut = cv2.resize(img_cut, (240, 80))
    img_label = cv2.rectangle(img, (xywh[0], xywh[1]),
                              (xywh[0] + xywh[2], xywh[1] + xywh[3]),
                              (0, 0, 255), 2)
    img_label = cv2.putText(img_label, cls, (xywh[0], xywh[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return img_cut, img_label


if __name__ == '__main__':
    device = select_device('0')
    yolo = DetectMultiBackend('weights/yolo.pt',
                              device=device,
                              data=ROOT / 'scripts/CCPD.yaml')
    path = "D:/Softwares/Python/CarPlateRecog-All-In-One/xxx.jpg"
    img = cv2.imread(path)
    # img = cv2.resize(img, (512, 512))
    img_cut = detect(img, yolo)
    img_cut = cv2.resize(img_cut, (240, 80))
    cv2.imshow('img', img_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
