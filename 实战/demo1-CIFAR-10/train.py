import torch
import torchvision

def train(device, model, dataloader, criterion, optimizer, epochs=25):
    model.to(device)
    model.train()

    for epoch in range(epochs):

        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)

            optimizer.zero_grad()

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), './pretrained/net.pth')


if __name__ == '__main__':
    pass