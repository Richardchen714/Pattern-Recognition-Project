'''
Pytorch自定义加载数据--自定义Dataset
https://blog.csdn.net/xuan_liu123/article/details/101145366
https://blog.csdn.net/wangnu_043/article/details/84797788

skimage.transform模块实现图片缩放
https://blog.csdn.net/scott198510/article/details/80548152

Dataset训练集、测试集划分
https://blog.csdn.net/sdnuwjw/article/details/111227327

Dataloader重要参数
https://blog.csdn.net/zyq12345678/article/details/90268668
Pytorch的nn.Conv2d（）详解
https://blog.csdn.net/qq_42079689/article/details/102642610

LeNet网络
https://blog.csdn.net/weixin_40197807/article/details/108689410 数据dataloader
https://blog.csdn.net/aass6d/article/details/106822386  网络模型
https://blog.csdn.net/qq_34392457/article/details/113748534 训练+测试
https://blog.csdn.net/m0_37867091/article/details/107136477 过程详解

错误net(input)  expected scalar type Double but found Float
https://blog.csdn.net/ttdxtt/article/details/112347283
错误dataset for i, data in enumerate(train_loader): TypeError: img should be PIL Image. Got ＜class ‘dict‘＞
https://blog.csdn.net/weixin_43793510/article/details/120564907 自定义dataset中 字典类型不能直接transfrom
错误DataLoader RuntimeError: stack expects each tensor to be equal size
https://www.cnblogs.com/happyamyhope/p/14957691.html 输入数据大小不一致∴调整图片大小
# author：张致宁
# date：2021/12/10
'''
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn as nn
import torch
from PIL import Image
from skimage import io, transform




class StrawData(Dataset):  # 继承Dataset
    def __init__(self, root_dir, transform=None):  # __init__是初始化该类的一些基础参数
        self.root_dir = root_dir  # 文件目录
        self.transform = transform  # 变换
        self.images = os.listdir(self.root_dir)  # 目录里的所有文件

    def __len__(self):  # 返回整个数据集的大小
        return len(self.images)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        image_index = self.images[index]  # 根据索引index获取该图片
        img_path = os.path.join(self.root_dir, image_index)  # 获取索引为index的图片的路径名
        img = io.imread(img_path)  # 读取该图片
        img = transform.resize(img, (64, 64)) # 重置大小为64×64
        img = np.array(img) # 转化成ndarray 方便tensor()转换
        # 字符分割_后.去除PNG,得到熟度，及标签 W35_10_ripe.PNG
        t = image_index.split('_')[2].split('.')[0]
        label = dict_ripe.get(t)
        if self.transform:
            img = self.transform(img)# 对样本进行变换
        sample = {'image': img, 'label': label}  # 根据图片和标签创建字典
        return sample  # 返回该样本


# 熟度对应标签字典
dict_ripe = {'unripe': 0, 'partripe': 1, 'ripe': 2}
BATCH_SIZE = 128 # 128张图片每
EPOCHS = 10 # 总共训练批次
root = 'D:/文档/东南大学本科/2021-2022学年/1学期/模式识别/SEU-PRProject/project1/Images/'
dataset=StrawData(root_dir=root,transform=transforms.ToTensor())
# 分割数据集80%训练 20%测试
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
# 随机分割
train_datasets, test_datasets = torch.utils.data.random_split(dataset, [train_size, test_size])
# 转化成Dataloader便于处理
# 训练集、测试集
train_loader = DataLoader(dataset=train_datasets,batch_size=10,shuffle=True)
test_loader = DataLoader(dataset=test_datasets, batch_size=test_size,shuffle=True)


class Models(torch.nn.Module): # 自定义模型
    '''
    卷积+激活+卷积+激活+池化+
    卷积+激活+卷积+激活+池化+
    展平
    全连接+激活+dropout（防止过拟合）+全连接-3类别
    '''
    def __init__(self):
        super(Models, self).__init__()
        self.Conv = torch.nn.Sequential(
            torch.nn.Conv2d(4, 64, kernel_size=(3, 3), padding=1), # PNG图片PGBA 通道数为4
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.Classes = torch.nn.Sequential(
            torch.nn.Linear(16 * 16 * 256, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(512, 3))

    def forward(self, inputs): # 正向传播过程
        x = self.Conv(inputs)
        x = x.view(-1, 16 * 16 * 256)
        x = self.Classes(x)
        return x


net = Models()  # 定义训练的网络模型
loss_function = nn.CrossEntropyLoss()  # 定义损失函数为交叉熵损失函数
optimizer = optim.Adam(net.parameters(), lr=0.001)  # 定义优化器（训练参数，学习率）
losses = []
for epoch in range(5):  # 一个epoch即对整个训练集进行一次训练
    running_loss = 0.0
    time_start = time.perf_counter()
    net.train()
    for step, data in enumerate(train_loader):  # 遍历训练集，step从0开始计算
        optimizer.zero_grad()  # 清除历史梯度
        # forward + backward + optimize
        outputs = net(data['image'].to(torch.float32))  # 正向传播
        loss = loss_function(outputs, data['label'])  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 优化器更新参数

        # 打印耗时、损失、准确率等数据
        running_loss += loss.item()
        losses.append(running_loss / 10)
        if step % 10 == 9:  # 每10步打印一次
            print("[{}] loss: {:.4f}".format(epoch, running_loss / 10))
            running_loss = 0.0
    # '''
    # 保存每一轮的模型
    model_name = 'classify-{}.pth'.format(epoch)
    torch.save(net, model_name)
    acc = 0.0
    count = 0
    # 加载模型
    test_net = torch.load(model_name)
    test_net.eval()
    for index, data in enumerate(test_loader):
        count += len(data['label'])
        outputs = test_net(data['image'].to(torch.float32))
        _, predict = torch.max(outputs, 1)
        acc += (data['label'] == predict).sum().item()
    # 计算准确率
    print("[{}] accurancy: {:.4f}".format(epoch, acc / count))
   # '''
print('训练完成')

# 保存训练得到的参数loss
save_path = './Lenet1.pth'
torch.save(net.state_dict(), save_path)
plt.plot(losses)
plt.show()
