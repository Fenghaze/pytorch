import torch.nn as nn
import torch
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def initialize_weights(self):
        for m in self.modules(): # 遍历网络的每一层，判断各层属于什么结构，对于不同的结构，设定不同的权值初始化方法
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data) # 对 w（权重）进行初始化
                if m.bias is not None: # 如果有 b（偏置），则 set b = 0
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()



if __name__ == '__main__':
    model  = Net()
    model.initialize_weights() # 初始化权重
    for parameter in model.state_dict():
        print(parameter, model.state_dict()[parameter].size())