from models import finetune, finetune1
import torch
from hy_dataset import ClassifyDataset, imshow
from torch import optim, nn
import time
import copy
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
# 单独训练
def train(device, model, trainloader, trainsize, criterion, optimizer, lr_decay, epochs=5):
    since = time.time()

    model.train()
    current_loss = 0.0
    current_acc = 0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs-1))
        print('-' * 20)

        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1) # preds 是每一行概率的最大值的下标

            loss = criterion(outputs, labels) # 每张图片的“平均？”损失
            loss.backward()
            optimizer.step()

            current_loss += loss.item() * inputs.size(0) # 累计每组 batch 的损失
            current_acc += (preds == labels).sum().item() # 预测准确的图片个数

        epoch_loss = current_loss / trainsize
        epoch_acc = current_acc.double() / trainsize

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            'Training', epoch_loss, epoch_acc))
        torch.save(model.state_dict(), './pretrained/bees&ants_epoch{}.pth'.format(epoch))
        lr_decay.step()  # 每循环完一个 epoch 就衰减学习率

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    print("Finished!")

# 训练+验证模型，保存最佳模型
def train_eval(device, model, dataloader, datasize, criterion, optimizer, lr_decay, epochs=25):
    since = time.time()
    writer = SummaryWriter('runs')
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            running_loss = 0.0
            running_corrects = 0
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # Iterate over data.
            for inputs, labels in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                running_corrects += (preds == labels).sum().item()
            if phase == 'train':
                lr_decay.step()

            print(phase, running_corrects,datasize[phase])

            epoch_loss = running_loss / datasize[phase] # train: 244, test: 153
            epoch_acc = running_corrects / datasize[phase]

            if phase == 'train':
                writer.add_scalar('Train Loss',epoch_loss, epoch)
                writer.add_scalar('Train Acc',epoch_acc, epoch)

            else:
                writer.add_scalar('Eval Loss',epoch_loss, epoch)
                writer.add_scalar('Eval Acc',epoch_acc, epoch)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    torch.save(best_model_wts, './pretrained/best_model.pth')
    # model.load_state_dict(best_model_wts)
    # return model


def visualize_model(device, model, evalloader, classes_name, num_images=6):
    model.load_state_dict(torch.load('./pretrained/best_model.pth'))
    was_training = model.training
    model.eval()
    fig = plt.figure()
    images_so_far = 0
    with torch.no_grad():
        for i, data in enumerate(evalloader):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)

            _, preds = torch.max(outputs, 1)

            for j in range(images.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted:{}'.format(classes_name[preds[j]]))
                imshow(images.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return

        model.train(mode=was_training)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = finetune1(device)
    root = '../data/hymenoptera_data' # 不用再加一个 '/'
    classes_name, dataset_sizes, dataloaders = ClassifyDataset(root)
    trainloader = dataloaders['train']
    trainsize = dataset_sizes['train']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    lr_decay = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) # 学习率调整
    #train(device, model, trainloader, trainsize, criterion,optimizer, lr_decay)

    train_eval(device, model, dataloaders, dataset_sizes, criterion,optimizer, lr_decay)
    #visualize_model(device,model,dataloaders['val'],classes_name)