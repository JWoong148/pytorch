import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm

# Hyper-parameters
batch_size = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using", device)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
mnist_train = datasets.MNIST(root='data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='data', train=False, download=True, transform=transform)
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)


class FCNN(nn.Module):
    def __init__(self, hidden_size=1024):
        super(FCNN, self).__init__()
        self.nn1 = nn.Linear(28 * 28, hidden_size)
        self.nn2 = nn.Linear(hidden_size, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.reshape(-1, 28 * 28)
        x = self.nn1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.nn2(x)
        return x


class CNN(nn.Module):
    def __init__(self, hidden_size=1024):
        super(CNN, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5, padding=2), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def train(model, train_loader, optimizer):
    model.train()

    for data, target in tqdm(train_loader, ncols=80):
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = F.cross_entropy(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += F.cross_entropy(output, target, reduction='sum').item()

            _, pred = output.data.max(1)
            correct += (pred == target).sum().item()
    test_loss /= len(test_loader.dataset)
    correct /= len(test_loader.dataset)
    return test_loss, correct


cnn = CNN()
if torch.cuda.is_available():
    cnn.cuda()


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


cnn.apply(init_weights)

optimizer = Adam(cnn.parameters(), lr=0.001)

num_epochs = 20
for epoch in range(num_epochs):
    loss = train(cnn, train_loader, optimizer)
    test_loss, correct = test(cnn, test_loader)
    print('Epoch [{}/{}] \tTrain loss: {:.4f} \tTest loss: {:.4f} \tTest acc: {:.2f}'.format(
        epoch + 1, num_epochs, loss, test_loss, correct * 100))
