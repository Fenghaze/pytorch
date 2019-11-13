
import torch
import torchvision
from mn_dataset import data, imshow
from models import Net


def test_grid(model, path, testloader, classes):
    model.load_state_dict(torch.load(path))
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    imshow(torchvision.utils.make_grid(images))
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(8)))


def test_all(model, path, testloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()
    correct = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            # ouputs.shape : [64, 10]  64 代表 batch_size，也是一组图片的样本数； 10 代表有 10 个输出标签，0-9 也对应各个数字
            _, predicted = torch.max(outputs.data, 1)
            """
            torch.max()的第一个输入是tensor格式，所以用outputs.data而不是outputs作为输入；
            第二个参数1是代表dim的意思，也就是取每一行的最大值
            返回输入张量给定维度上每行的最大值，并同时返回每个最大值的位置索引
            """
            correct += (predicted == labels).sum().item()
            """
            batch_size = 64
            predicted 是一组图片的 64 个下标，也是各个数字图片的预测值
            correct += (predicted == labels).sum().item() 计算预测值与真值相同的个数，即准确个数
            """
    print('Accuracy of the network [{}/{}]({:.0f}%)' .format (correct, len(testloader.dataset),
            100. * correct / len(testloader.dataset)))


if __name__ == '__main__':
    model = Net()
    root = '../data'
    path = './pretrained/mn_net.pth'
    _, testloader, classes = data(root)
    test_all(model,path, testloader)