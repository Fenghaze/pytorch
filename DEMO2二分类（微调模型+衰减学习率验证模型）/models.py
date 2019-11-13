import torch.nn as nn
from torchvision import models
import torch
from tensorboardX import SummaryWriter

def finetune(device):
    model_ft = models.resnet18(pretrained=True) # 选择预训练模型
    num_ftrs = model_ft.fc.in_features # 全连接层的输入特征数

    model_ft.fc = nn.Linear(num_ftrs, 2) # 修改最后的全连接层，二分类问题，输出 2 个预测值

    model = model_ft.to(device)

    return model


def finetune1(device):
    model_conv = models.resnet18(pretrained=True)
    for parameter in model_conv.parameters():
        parameter.requires_grad = False

    num_ftrs = model_conv.fc.in_features

    model_conv.fc = nn.Linear(num_ftrs, 2)
    model = model_conv.to(device)

    return model

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = finetune1(device)
    for parameter in model.state_dict():
        print(parameter, '\t', model.state_dict()[parameter].size())
    writer = SummaryWriter('runs')
    writer.add_graph(model, torch.randn(6, 3, 224, 224).to(device))
