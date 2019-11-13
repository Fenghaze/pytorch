import torch.nn
from models import Net
from mn_dataset import data
from utils import plot_curve

def train(model, dataloader, criterion, optimizer, epochs=5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_loss = []
    model.train()
    for epoch in range(epochs):
        current_loss = 0.0 # 一次 epoch 的损失值

        for batch_idx, data in enumerate(dataloader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)

            optimizer.zero_grad()

            loss = criterion(outputs, labels) # 一个 batch 的损失值
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            current_loss += loss.item()

            """
            len(data):2
            len(dataloader):938 个 batches （batch_size = 64, 64*938 = 60032）
            lend(dataloader.dataset):60000 样本数
            """
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(data), len(dataloader.dataset),
                           100. * batch_idx / len(dataloader), loss.item()))

    save_path = './pretrained/mn_net.pth'
    torch.save(model.state_dict(), save_path)
    print("Finished!")
    plot_curve(train_loss)

if __name__ == '__main__':
    root = '../data'
    trainloader, _, _ = data(root)
    model = Net()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    train(model, trainloader, criterion, optimizer)