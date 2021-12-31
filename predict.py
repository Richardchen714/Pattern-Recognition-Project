# 导入包
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
from PIL import Image
from LeNet import StrawData

#数据预处理
root = 'D:/文档/东南大学本科/2021-2022学年/1学期/模式识别/SEU-PRProject/project1/tests/'
dataset=StrawData(root_dir=root,transform=transforms.ToTensor())
test = DataLoader(dataset=dataset,batch_size=len(dataset),shuffle=True)

# 预测
test_net = torch.load('classify-4.pth')
test_net.eval()
for index, data in enumerate(test):
    with torch.no_grad():
        outputs = test_net(data['image'].to(torch.float32))
        _, predict = torch.max(outputs, 1)
        print("image:{} , predict:{}".format(data['label'], predict))

