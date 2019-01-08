import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Hyper-parameters
batch_size = 64

# Use cuda if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using", device)

# Load data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
mnist = datasets.MNIST(root='../data', download=True, transform=transform)
data_loader = DataLoader(mnist, batch_size=batch_size, shuffle=True)

# Xavier initialization
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
                # [?, 100] -> [?, 256]
                nn.Linear(100, 256),  nn.LeakyReLU(0.2),
                nn.Linear(256, 512),  nn.LeakyReLU(0.2), 
                nn.Linear(512, 1024), nn.LeakyReLU(0.2),
                nn.Linear(1024, 28 * 28), 
                nn.Tanh())

        if torch.cuda.is_available(): self.cuda()

    def forward(self, x):
        return self.main(x).view(-1, 1, 28, 28)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
                nn.Linear(28 * 28, 1024), nn.LeakyReLU(0.2), nn.Dropout(), 
                nn.Linear(1024, 512), nn.LeakyReLU(0.2), nn.Dropout(),
                nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Dropout(),
                nn.Linear(256, 1),
                nn.Sigmoid())

        if torch.cuda.is_available(): self.cuda()

    def forward(self, x):
        return self.main(x.view(-1, 28 * 28))

# Define models
G = Generator()
D = Discriminator()

G_optimizer = Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
D_optimizer = Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Binary Cross Entropy loss
criterion = nn.BCELoss()

# Apply Xavier initialization on weights
G.apply(init_weights)
D.apply(init_weights)

num_epochs = 100
num_test_samples = 16
test_z = torch.randn((num_test_samples, 100)).to(device)

for epoch in range(num_epochs):
    for i, data in enumerate(tqdm(data_loader, ncols=80)):
        real_data, _ = data
        batch_size = len(real_data)

        # Train discriminator
        real_data = real_data.to(device)

        target_real = torch.ones(batch_size, 1).to(device)
        target_fake = torch.zeros(batch_size, 1).to(device)

        D_real = D(real_data)
        D_loss_real = criterion(D_real, target_real)

        z = torch.randn((batch_size, 100)).to(device)
        fake_data = G(z)

        D_fake = D(fake_data)
        D_loss_fake = criterion(D_fake, target_fake)

        D_loss = D_loss_real + D_loss_fake

        D.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        # Train generator
        z = torch.randn((batch_size, 100)).to(device)
        fake_data = G(z)

        D_fake = D(fake_data)
        G_loss = criterion(D_fake, target_real)

        G.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        # if (i + 1) % 100 == 0:
        #     print('Epoch [{}/{}] Batch [{}/{}]\nD loss: {:.4f} \tG loss: {:.4f}'.format(
        #         epoch + 1, num_epochs, i + 1, len(data_loader), D_loss, G_loss))

    # Print result
    test_images = G(test_z)
    fig = plt.figure(figsize=(4, 4))
    fig.suptitle('Epoch {}'.format(epoch + 1))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=.05, hspace=.05)

    for i, img in enumerate(test_images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        plt.imshow(img.data[0])
    plt.show()
    # print('Epoch [{}/{}] \tDiscriminator loss: {:.4f} \tGenerator loss: {:.4f} \nD(x): {:.4f} \tD(G(z)): {:.4f}'.format(
    # epoch + 1, num_epochs, D_loss, G_loss, D_real, D_fake))
