from ci_dataset import imshow
import torchvision
from ci_dataset import data
import torch
from models import Net

def gt(testloader, classes): # 查看真值
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))
    print('GT:', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

def test_grid(path, model, testloader, classes):
    model.load_state_dict(torch.load(path)) # 加载 pth 文件
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

def test_all(path, model, testloader, classes):
    model.load_state_dict(torch.load(path)) # 加载 pth 文件
    correct = 0
    total = 0
    with torch.no_grad(): # 关闭自动求导
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0) # 取第 0 位的数字
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze() # 压缩维度

            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

if __name__ == '__main__':
    model = Net()
    _, testloader, classes = data()
    path = './pretrained/cifar_net.pth'
    test_all(path, model, testloader, classes)