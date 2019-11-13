import torch
import torch.optim as optim
from models import Net
import torch.nn as nn
from ci_dataset import data

def train(model, dataloader, optimizer, criterion, epochs=2):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    for epoch in range(epochs):

        running_loss = 0.0

        for epoch in range(2):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(dataloader, 0): # enumerate(iteration, start)
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

    path = './pretrained/cifar_net.pth'
    torch.save(model.state_dict(), path)
    print('Finished Training')


if __name__ == '__main__':
    model = Net()
    trainloader, testloader, classes = data()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    train(model, trainloader, optimizer, criterion )