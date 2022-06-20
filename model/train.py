#构建模型
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

#加载数据集
from data.dataset import HSI_Loader







# 多层感知机网络
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.l1 = nn.Linear(176,120)
#         self.l2 = nn.Linear(120, 80)
#         self.l3 = nn.Linear(80, 40)
#         self.l4 = nn.Linear(40, 5)
#
#
#
#     def forward(self, x):
#
#         x = x.view(-1, 176)
#         x = F.relu(self.l1(x))
#         x = F.relu(self.l2(x))
#         x = F.relu(self.l3(x))
#
#         return self.l4(x)
#         #return self.l5(x)
from model import Encode_Net

"""训练"""
def train_net(epochs,batch_size,net,device,train_dataset,lr=0.00001,):


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    criterion = torch.nn.CrossEntropyLoss()
    # 神经网络已经逐渐变大，需要设置冲量momentum=0.5
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    best_loss = float('inf')
    train_acc = 0
    n = 0
    # 训练epochs次
    for epoch in range(epochs):
        # 训练模式
        net.train()
        # 按照batch_size开始训练
        for curve, label in train_loader:
            optimizer.zero_grad()
            # 将数据拷贝到device中
            curve = curve.reshape(len(curve), 1, -1).to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.long)
            # 使用网络参数，输出预测结果
            out = net(curve)

            loss = criterion(out, label)
            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'best_model_net.pth')
            # 更新参数
            loss.backward()
            optimizer.step()
            train_acc += torch.eq(torch.max(out, dim=1).indices,label).sum()
            n += label.shape[0]

            print(f'epoch:{epoch}, loss:{loss.item()},train_acc:{train_acc/n}')
            print(f'best_loss:{best_loss.item()}')


if __name__ == '__main__':
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，曲线单通道1，分类为2。
    net = Encode_Net()
    # 将网络拷贝到deivce中
    net.to(device=device)
    dataset = HSI_Loader(r'D:\ZPEAR\SUTDF-master\SUTDF-master\Class\train_data\train_data.npy',
                         r'D:\ZPEAR\SUTDF-master\SUTDF-master\Class\train_data\label_data.npy')


    train_net(400, 512,net,device,dataset)