# utils.py
import torch
from matplotlib import pyplot as plt


def plot_curve(data):  # 下降曲线的绘制
    fig = plt.figure()
    plt.plot(range(len(data)), data, color='blue')  #
    plt.legend(['value'], loc='upper right')  #
    plt.xlabel('step')
    plt.ylabel('value')
    plt.show()


def plot_image(img, label, name):  # 画图片，帮助看识别结果
    fig = plt.figure()
    for i in range(6):  # 6个图像，两行三列
        # print(i) 012345
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()  # 紧密排版
        plt.imshow(img[i][0] * 0.3081 + 0.1307, cmap='gray', interpolation='none')
        # 均值是0.1307，标准差是0.3081，

        plt.title("{}:{}".format(name, label[i].item()))
        # name:image_sample   label[i].item():数字

        plt.xticks([])
        plt.yticks([])
    plt.show()


def one_hot(label, depth=10):
    out = torch.zeros(label.size(0), depth)
    idx = torch.LongTensor(label).view(-1, 1)
    out.scatter_(dim=1, index=idx, value=1)
    return out