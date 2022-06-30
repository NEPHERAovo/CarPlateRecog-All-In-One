import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

import cnnforcp

#超参数(Hyperparameters)
batch_size = 32
learning_rate = 1e-2
num_epoches = 5

data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), transforms.Grayscale(num_output_channels=1)])

#下载训练集MNIST手写数字训练集
train_dataset = datasets.ImageFolder('D:\matlab\license_plate_system/bptest\dataset/nuwchepai/',  transform=data_tf)
test_dataset  = datasets.ImageFolder('D:\matlab\license_plate_system/bptest\dataset/nuwchepai/',  transform=data_tf)
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader   = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#print(test_dataset.class_to_idx)
#print(test_dataset.imgs)


model = cnnforcp.chepai_CNN(1)
if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

#训练数据
for epoch in range(num_epoches):
    train_loss = 0
    train_acc = 0
    model.train() #开启训练模式，即启用batch normalization和drop out
    for data in train_loader: #data为train_loader中的一个批次样本
        img, label = data #img维数为[32, 1, 48, 32]，cnn网络img保持原有维数不变
        #print(img.mode());
        if torch.cuda.is_available():
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)
        #==========forward====================
        out = model(img)
        loss = criterion(out, label)
        #==========backward===================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #=========record loss=================
        train_loss += loss.data/len(train_dataset)
        #计算分类的准确率
        #在分类问题中，通常需要使用max()函数对softmax函数的输出值进行操作，求出预测值索引
        #torch.max(input, dim), 其中input是softmax函数输出的一个tensor,
        #dim是max函数索引的维度0/1，0是每列的最大值，1是每行的最大值
        #函数会返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引。
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        train_acc += num_correct.cpu().numpy()/len(train_dataset)

    #输出阶段训练结果
    print('*'*10)
    print('epoch: {}, train loss: {:.4f}, train acc: {:.4f}'.format(epoch+1, train_loss, train_acc))
torch.save(model, 'weights/cpmodelnew.pkl')   #保存训练好的网络为pkl文件



#测试数据
#model.load_state_dict()
model.eval() #让model变为测试模式，网络会沿用batch normalization的值，但不使用drop out
eval_loss = 0
eval_acc = 0
with torch.no_grad():
    for data in test_loader:
        img, label = data
        if torch.cuda.is_available():
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)
        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.data * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred==label).sum()
        eval_acc += num_correct.data
print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss.cpu().numpy()/(len(test_dataset)), eval_acc.cpu().numpy()/(len(test_dataset))))


'''
from torchvision import transforms as T
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder


dataset = ImageFolder('dataset/nuwchepai/')

# cat文件夹的图片对应label 0，dog对应1
print(dataset.class_to_idx)

# 所有图片的路径和对应的label
print(dataset.imgs)

# 没有任何的transform，所以返回的还是PIL Image对象
#print(dataset[0][1])# 第一维是第几张图，第二维为1返回label
#print(dataset[0][0]) # 为0返回图片数据
plt.imshow(dataset[0][0])
plt.axis('off')
plt.show()
'''