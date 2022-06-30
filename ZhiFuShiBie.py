import numpy as np
import torch
import numpy
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import os
import cnnforcp
import cv2

wordlist = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E",
    "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "桂", "沪", "津", "晋", "京", "陕", "苏", "皖", "湘", "豫", "粤",
    "浙"
]
chepailist = []
tupianlist = []

batch_size = 1
learning_rate = 1e-2
num_epoches = 5

# data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), transforms.Grayscale(num_output_channels=1)])
data_tf = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

#state_dict = torch.load('cpmodel.pth')
#model = cnnforcp.chepai_CNN(1)
#model.load_state_dict(state_dict)
#model = torch.load('cpmodelnew.pkl')  #加载训练好的网络
model = torch.load('weights/cpmodelnew.pkl')  #加载训练好的网络

if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

model.eval()  #让model变为测试模式，网络会沿用batch normalization的值，但不使用drop out


# img = cv2.imread(r"D:\matlab\license_plate_system\bptest\dataset\xztpdefg\0\7.jpg")
def danGeZiFuShiBie(img):
    with torch.no_grad():
        img = cv2.resize(img, (32, 48))
        img = data_tf(img)
        img = img.reshape(1, 1, 48, 32)
        if torch.cuda.is_available():
            img = Variable(img).cuda()
        else:
            img = Variable(img)
        outputs = model(img)
        _, indices = torch.max(outputs, 1)
        indices = indices.cpu().numpy()

    return wordlist[int(indices)]
    #print(wordlist[int(indices)])


# danGeZiFuShiBie(img)