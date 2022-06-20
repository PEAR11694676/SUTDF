import torch.nn.functional as F
import torch

from data.dataset import HSI_Loader
from parts import * # 运行train时改为".parts"，model时候改为"parts"，路径问题
import sys
sys.path.append("..")

class Encode_Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.down1 = Conv(1, 16)
        self.down2 = Conv(16, 32)
        self.down3 = Conv(32,64)
        self.fc = FC()

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.fc(x3)

        return x4


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Encode_Net()
    net.to(device=device)
    dataset = HSI_Loader(r'D:\ZPEAR\SUTDF-master\SUTDF-master\Class\train_data\train_data.npy',
                         r'D:\ZPEAR\SUTDF-master\SUTDF-master\Class\train_data\label_data.npy')
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=32,
                                               shuffle=False)
    batch_size = 32
    for curve,label in train_loader:
        curve = curve.reshape(batch_size, 1, -1).to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.float32)
        break
    print(curve.shape)
    # 使用网络参数，输出预测结果
    h = net(curve)
    print(h.shape)

