import torch
import matplotlib.pyplot as plt


def plot(pred, acc,h):
    img = pred.indices.resize(100, 100)
    img = torch.squeeze(img).detach().cpu().numpy()
    plt.imshow(img)
    plt.title(f'Target Classfication{h}m')
    plt.text(x=33,  # 文本x轴坐标
             y=95,  # 文本y轴坐标
             s=f'准确率为：{acc}',  # 文本内容

             color='r',

             # 添加文字背景色
             bbox={'facecolor': '#74C476',  # 填充色
                   'edgecolor': 'b',  # 外框色
                   'alpha': 0.5,  # 框透明度
                   'pad': 8,  # 本文与框周围距离
                   }

             )
    plt.legend()
    plt.show()