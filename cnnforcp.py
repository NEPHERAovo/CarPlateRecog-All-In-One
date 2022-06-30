import torch.nn as nn

class chepai_CNN(nn.Module):
    def __init__(self, c):
        super(chepai_CNN, self).__init__()
        self.layer1 = nn.Sequential( #该函数将按照参数传递的顺序将其依次添加到处理模块中
            #nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,[dilation, groups, bias])，各参数如下：
            #in_channels对应输入数据体的深度
            #out_channels对应输出数据体的深度（特征图的数量），该参数定义滤波器（卷积核）的数量。
            #kernel_size表示滤波器(卷积核)的大小，使用一个数字表示相同高宽的卷积核，或不同数字表示高宽不等的卷积核，如kernel_size=(3,2)
            #stride表示滑动步长，默认stride=1
            #padding为周围0填充行数，padding=0(默认)为不填充
            #bias是一个布尔值，默认bias=True，表示使用偏移置
            #dilation表示卷积对于输入数据体的空间间隔，默认dilation=1
            #groups表示输出数据体深度上和输入数据体深度上的联系，默认groups=1,也就是所有输出和输入都是关联的。
            nn.Conv2d(c, 16, kernel_size=3), #输出特征图个数(深度)16,原大小48*32，现46*30
            nn.BatchNorm2d(16), #Batch Normalization强行将数据拉回到均值为0，方差为1的正太分布上，参数为特征图的数量
            nn.ReLU(inplace=True) #inplace=True节省内(显)存空间，省去反复申请和释放内存的时间，但会对输入的变量进行覆盖，类似地址传递
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3), #输出特征图个数为32,大小44*28
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) #输出特征图个数为32,大小22*14
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3), #输出特征图个数为64,大小20*12
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3), #输出特征图个数为128,大小18*10
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  #输出特征图个数为128,大小9*5
        )

        self.fc = nn.Sequential(
            nn.Linear(128*9*5, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 47)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x