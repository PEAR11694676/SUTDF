import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score,precision_score,recall_score,cohen_kappa_score, accuracy_score
from data.dataset import HSI_Loader
from model import  Encode_Net
import torch.nn as nn
from plot_ROC import ROC_curve
from train import train_net
from utils.plot_pred import plot

plt.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']  # 汉字字体,优先使用楷体，如果找不到楷体，则使用黑体
plt.rcParams['font.size'] = 12  # 字体大小
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
torch.set_printoptions(precision=4)


def pred_net(net,device, curve,label, batch_size=25):
    # 加载训练集
    global recall_score
    dataset = HSI_Loader(curve,label)

    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=10000,
                                               shuffle=False)
    accuracy = 0
    n = 0
    criterion = torch.nn.CrossEntropyLoss()
    # 测试模式
    # 加载模型参数
    net.load_state_dict(torch.load('best_model_net.pth', map_location=device))# 加载模型参数
    net.eval()
    dict = {0:'water',1:'Iron',2:'Nylon_Carpet',3:"Plastic_PETE",4:"Wood_Beam"}
    for curve, label in train_loader:
        # 将数据拷贝到device中
        curve = curve.reshape(len(curve), 1, -1).to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.long)

        # 使用网络参数，输出预测结果
        out = net(curve)

        # 计算loss
        loss = criterion(out, label)
        pred = torch.max(out, dim=1)


        print(f'loss:{loss}')
        print(f'label:{label}')
        print(f'pred；{pred.indices}')
        # print(f'out:{out}')

        accuracy += torch.eq(pred.indices,label).sum()
        n += label.shape[0]
        acc = accuracy / n
        print(f'准确率为：{acc}')

        f1score = f1_score(label.cpu().numpy(), pred.indices, average='micro')
        print(f'f1_score:{f1score}')

        precesion_score = precision_score(label.cpu().numpy(), pred.indices, average='micro')
        print(f'precesion_score:{precesion_score}')

        recall_score = recall_score(label.cpu().numpy(), pred.indices, average='micro')
        print(f'recall_score:{recall_score}')

        accuracy_score = accuracy_score(label.cpu().numpy(), pred.indices)
        print(f'accuracy_score:{accuracy_score}')





        # plot(pred,acc,0.5)
        # np.save('../data/ROC_data/pred_4m.npy',pred.indices.cpu().numpy())

        # plt.legend(['water','Iron','Nylon_Carpet',"Plastic_PETE","Wood_Beam"])



        # for i in range(curve.size()[0]):
        #     print(f"判定为:{dict[pred.indices[i].item()]}")

        wavelength = [398.10, 401.30, 404.50, 407.60, 410.80, 414.00, 417.20, 420.40, 423.60, 426.80, 430.00,
                      433.30, 436.50, 439.70, 442.90, 446.10, 449.40, 452.60, 455.80, 459.10, 462.30, 465.50,
                      468.80, 472.00, 475.30, 478.50, 481.80, 485.10, 488.30, 491.60, 494.90, 498.10, 501.40,
                      504.70, 508.00, 511.30, 514.60, 517.80, 521.10, 524.40, 527.70, 531.00, 534.40, 537.70,
                      541.00, 544.30, 547.60, 550.90, 554.30, 557.60, 560.90, 564.30, 567.60, 570.90, 574.30,
                      577.60, 581.00, 584.30, 587.70, 591.10, 594.40, 597.80, 601.20, 604.50, 607.90, 611.30,
                      614.70, 618.10, 621.40, 624.80, 628.20, 631.60, 635.00, 638.40, 641.80, 645.30, 648.70,
                      652.10, 655.50, 658.90, 662.40, 665.80, 669.20, 672.70, 676.10, 679.50, 683.00, 686.40,
                      689.90, 693.30, 696.80, 700.20, 703.70, 707.20, 710.60, 714.10, 717.60, 721.10, 724.60,
                      728.00, 731.50, 735.00, 738.50, 742.00, 745.50, 749.00, 752.50, 756.00, 759.50, 763.10,
                      766.60, 770.10, 773.60, 777.10, 780.70, 784.20, 787.80, 791.30, 794.80, 798.40, 801.90,
                      805.50, 809.00, 812.60, 816.20, 819.70, 823.30, 826.90, 830.40, 834.00, 837.60, 841.20,
                      844.80, 848.40, 852.00, 855.60, 859.20, 862.80, 866.40, 870.00, 873.60, 877.20, 880.80,
                      884.40, 888.10, 891.70, 895.30, 899.00, 902.60, 906.20, 909.90, 913.50, 917.20, 920.80,
                      924.50, 928.10, 931.80, 935.50, 939.10, 942.80, 946.50, 950.20, 953.80, 957.50, 961.20,
                      964.90, 968.60, 972.30, 976.00, 979.70, 983.40, 987.10, 990.80, 994.50, 998.20, 1002.00]
        # for i in range(curve.size()[0]):
        #     l = pred.indices[i]
        #     plt.figure()
        #     plt.plot(wavelength, torch.squeeze(curve[i,:]).detach().cpu().numpy(),
        #              label='reconstruct', color='r', marker='o', markersize=3)

        #     plt.xlabel('band')
        #     plt.ylabel('reflect value')
        #     plt.legend(['true_r','pred_r'])







if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Encode_Net()
    # 将网络拷贝到deivce中
    net.to(device=device)
    curve = '../data/test_data/Mix_target_0.5m.npy'
    label = '../data/test_data/Mix_target_0.5m_label.npy'




    pred_net(net,device, curve,label)
