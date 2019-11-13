import torch
import torchvision
from mymodels import Net
from fa_dataset import FaDataset, imshow
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

def train(device, model, dataloader, criterion, optimizer, epochs=30, writer=None):
    model.to(device)
    model.train()

    for epoch in range(epochs):

        current_loss = 0.0
        current_correct = 0

        for i, data in enumerate(dataloader, 0):
            """
            len(dataloader) 968; 
            len(dataloader.datasets) 60000 ≈ 938 * 64
            
            968 个内循环，每次内循环处理 64 张图片
            """
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            optimizer.zero_grad()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            current_loss += loss.item()
            current_correct += (labels == preds).sum().item()

            # if i % 100 == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch+1, i * len(data), len(dataloader.dataset),
            #                100. * i / len(dataloader), loss.item()))

        print(current_loss, current_correct, len(dataloader))
        epoch_loss = current_loss / len(dataloader.dataset) # 一次 epoch 的 loss 值 / 总图片数 = 每个图片的损失值
        epoch_acc = current_correct / len(dataloader.dataset) # 一次 epoch 的准确个数 / 总图片数 = 精确度
        writer.add_scalar('Test Loss', epoch_loss, epoch)
        writer.add_scalar('Test Acc', epoch_acc, epoch)

        print('Train Epoch [{}/{}]'.format(epoch+1, epochs))
        print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    print('Finished Train!')
    torch.save(model.state_dict(), './pretrained/net.pth')


# def eval(device, model, dataloader, criterion, optimizer, epochs=30, writer=None):
#     model.to(device)
#     model.eval()
#
#     for epoch in range(epochs):
#
#         current_loss = 0.0
#         current_correct = 0
#
#         for i, data in enumerate(dataloader, 0):
#             images, labels = data
#             images = images.to(device)
#             labels = labels.to(device)
#             optimizer.zero_grad()
#             with torch.no_grad():
#                 outputs = model(images)
#                 loss = criterion(outputs, labels)
#                 _, preds = torch.max(outputs, 1)
#                 current_loss += loss.item()
#                 current_correct += (labels == preds).sum().item()
#
#             # if i % 100 == 0:
#             #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#             #         epoch+1, i * len(data), len(dataloader.dataset),
#             #                100. * i / len(dataloader), loss.item()))
#
#         print(current_loss, current_correct, len(dataloader))
#         epoch_loss = current_loss / len(dataloader.dataset)  # 一次 epoch 的 loss 值 / 总图片数 = 每个图片的损失值
#         epoch_acc = current_correct / len(dataloader.dataset)  # 一次 epoch 的准确个数 / 总图片数 = 精确度
#         writer.add_scalar('Eva Loss', epoch_loss, epoch)
#         writer.add_scalar('Eva Acc', epoch_acc, epoch)
#         print('Eval Epoch [{}/{}]'.format(epoch + 1, epochs))
#         print('Eval Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
#
#     print('Finished Eval!')


def visualize(train_loss, eval_loss):
    pass

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net()
    trainloader, testloader, classes_name = FaDataset()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    writer = SummaryWriter('runs')

   # train(device, model, trainloader, criterion, optimizer, epochs=30, writer=writer)
    eval(device, model, testloader, criterion, optimizer, epochs=30, writer=writer)
