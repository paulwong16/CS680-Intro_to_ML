import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


class vgg11(nn.Module):

    def __init__(self):
        super(vgg11, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv3_bn = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv4_bn = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv5_bn = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv6_bn = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1_bn(self.conv1(x))), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2_bn(self.conv2(x))), (2, 2))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.max_pool2d(F.relu(self.conv4_bn(self.conv4(x))), (2, 2))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.max_pool2d(F.relu(self.conv6_bn(self.conv6(x))), (2, 2))
        x = F.relu(self.conv6_bn(self.conv6(x)))
        x = F.max_pool2d(F.relu(self.conv6_bn(self.conv6(x))), (2, 2))
        x = x.squeeze(2)
        x = x.squeeze(2)
        x = F.dropout(F.relu(self.fc1(x)), 0.5)
        x = F.dropout(F.relu(self.fc2(x)), 0.5)
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    # If false, use original dataset, else use agumentation.
    agumentation = True
    # if not exist, download mnist dataset
    train_set = datasets.MNIST(root='./data/', train=True, download=True, transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))
    test_set = datasets.MNIST(root='./data/', train=False, download=True, transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=512, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=512)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # augmentation
    if agumentation:
        train_set_H = datasets.MNIST(root='./data/', train=True, download=True,
                                    transform=transforms.Compose([transforms.Resize((32, 32)),
                                                                transforms.RandomHorizontalFlip(p=1),
                                                                transforms.ToTensor()]))
        train_set_V = datasets.MNIST(root='./data/', train=True, download=True,
                                     transform=transforms.Compose([transforms.Resize((32, 32)),
                                                                   transforms.RandomVerticalFlip(p=1),
                                                                   transforms.ToTensor()]))
        t = transforms.Lambda(lambda x: x + 1.0 * torch.randn_like(x))
        train_set_G = datasets.MNIST(root='./data/', train=False, download=True,
                                    transform=transforms.Compose([transforms.Resize((32, 32)),
                                                                    transforms.ToTensor(), t]))
        train_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([train_set, train_set_H, train_set_G, train_set_V]), batch_size=512, shuffle=True)

    model = vgg11()
    # print(model)
    if torch.cuda.is_available():
        model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    # Training and testing
    for epoch in range(1, 6):
        model.train()
        train_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        accuracy = correct / len(train_loader.dataset)
        train_accuracies.append(accuracy)
        print('Train Epoch: %i, Loss: %f, Accuracy: %f' % (epoch, train_loss, accuracy))
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
                loss = criterion(output, target)
                test_loss += loss.item()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        accuracy = correct / len(test_loader.dataset)
        test_accuracies.append(accuracy)
        print('Test Epoch: %i, Loss: %f, Accuracy: %f' % (epoch, test_loss, accuracy))

    plt.figure()
    epochs = [i for i in range(1,6)]
    plt.plot(epochs, train_losses)
    plt.xlabel('epochs')
    plt.ylabel('train loss')
    plt.show()
    plt.figure()
    epochs = [i for i in range(1, 6)]
    plt.plot(epochs, test_losses)
    plt.xlabel('epochs')
    plt.ylabel('test loss')
    plt.show()
    plt.figure()
    epochs = [i for i in range(1, 6)]
    plt.plot(epochs, train_accuracies)
    plt.xlabel('epochs')
    plt.ylabel('train accuracy')
    plt.show()
    plt.figure()
    epochs = [i for i in range(1, 6)]
    plt.plot(epochs, test_accuracies)
    plt.xlabel('epochs')
    plt.ylabel('test accuracy')
    plt.show()

    # Now test filp 1
    test_set = datasets.MNIST(root='./data/', train=False, download=True,
                              transform=transforms.Compose([transforms.Resize((32, 32)),
                                                            transforms.RandomHorizontalFlip(p=1),
                                                            transforms.ToTensor()]))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=512)
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            loss = criterion(output, target)
    accuracy = correct / len(test_loader.dataset)
    print('After horizon flip, test accuracy: %f' % accuracy)

    # Now test flip 2
    test_set = datasets.MNIST(root='./data/', train=False, download=True,
                              transform=transforms.Compose([transforms.Resize((32, 32)),
                                                            transforms.RandomVerticalFlip(p=1),
                                                            transforms.ToTensor()]))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=512)
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            loss = criterion(output, target)
    accuracy = correct / len(test_loader.dataset)
    print('After vertical flip, test accuracy: %f' % accuracy)

    # Now test Gaussian Noise
    for sigma in [0.1, np.sqrt(0.1), 1]:
        t = transforms.Lambda(lambda x: x + sigma * torch.randn_like(x))
        test_set = datasets.MNIST(root='./data/', train=False, download=True,
                                  transform=transforms.Compose([transforms.Resize((32, 32)),
                                                                transforms.ToTensor(), t]))
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=512)
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
                loss = criterion(output, target)
        accuracy = correct / len(test_loader.dataset)
        print('After adding %f gaussian noise, test accuracy: %f' % (sigma, accuracy))



