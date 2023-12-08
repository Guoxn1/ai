import torch

torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 200

device="cuda"



class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims,condition_size):
        super(VariationalEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7+condition_size, 512)
        self.fc_mu = nn.Linear(512, latent_dims)
        self.fc_logvar = nn.Linear(512, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc
        self.N.scale = self.N.scale
        self.kl = 0


    def forward(self, x,c):
        
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.cat((x, c), dim=1)
        x = F.relu(self.fc1(x))
        
        mu = self.fc_mu(x).to(device)
        logvar = self.fc_logvar(x).to(device)
        sigma = torch.exp(0.5 * logvar)

        sam = self.N.sample(mu.shape).to(device)
        z = mu + sigma * sam

        self.kl = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())).sum()
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dims,condition_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dims+condition_size, 512)
        self.fc2 = nn.Linear(512, 7 * 7 * 64)
        self.conv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)
    def forward(self, z,c):
        z = torch.cat((z, c), dim=1)
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = z.view(-1, 64, 7, 7)  # Reshape to 4D tensor
        
        z = F.relu(self.conv1(z))
        reconstructed_img = torch.sigmoid(self.conv2(z))
        
        return reconstructed_img

class VariationalAutoencoder(nn.Module):
    def __init__(self,latent_dims,condition_size):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims,condition_size).to(device)
        self.decoder = Decoder(latent_dims,condition_size).to(device)

    def forward(self, x,c):
        
        z = self.encoder(x, c)
        return self.decoder(z, c)


def train(autoencoder, data, epochs=10):
    device="cuda"
    autoencoder = autoencoder.to(device)
    opt = torch.optim.Adam(autoencoder.parameters())
    for epoch in range(epochs):
        for x, y in data:
            x = x.to(device)  # GPU
            y = F.one_hot(y.to(device),condition_size)
            x_hat = autoencoder(x,y)
            loss = ((x - x_hat) ** 2).sum() + autoencoder.encoder.kl
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(epoch)
    return autoencoder



def plot_pre_imgs(cvae,latent_dims):

    sample = torch.randn(1,latent_dims).to(device)
    fig, ax = plt.subplots(2, 5, figsize=(10, 4))
    n = 0
    for i in range(2):
        for j in range(5):
            i_number = n*torch.ones(1).long().to(device)
            condit = F.one_hot(i_number,condition_size)#将数字进行onehot encoding
            gen = cvae.decoder(sample,condit)[0].view(28,28)#生成
            n = n+1
            ax[i, j].imshow(gen.cpu().detach().numpy(), cmap='gray')
            ax[i, j].axis('off')

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()


if __name__ == "__main__":
    # device="cpu"
    condition_size=10
    latent_dims = 8
    cvae = VariationalAutoencoder(latent_dims,condition_size)  # GPU
    data = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data',
                                   transform=torchvision.transforms.ToTensor(),
                                   download=True),
        batch_size=128,
        shuffle=True)
    
    
    cvae = train(cvae, data,epochs=10)
    plot_pre_imgs(cvae,latent_dims)
    plt.show()
    torch.save(cvae,"cnn_cvae.pth")