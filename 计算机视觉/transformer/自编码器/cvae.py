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
    def __init__(self, input_size,hidden_size,latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, latent_dims)
        self.linear3 = nn.Linear(hidden_size, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc  # hack to get sampling on the GPU
        self.N.scale = self.N.scale
        self.kl = 0

    def forward(self, x,c):
        x = torch.flatten(x, start_dim=1)
        x = torch.cat((x,c),dim=1)
        x = F.relu(self.linear1(x))
        mu = self.linear2(x).to(device)
        sigma = torch.exp(self.linear3(x)).to(device)
        sam = self.N.sample(mu.shape).to(device)
        z = mu + sigma * sam
        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dims,hidden_size,output_size):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, z,c):
        z = torch.cat((z,c),dim=1)
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, 28, 28))


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_size,output_size,hidden_size,latent_dims,condition_size):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(input_size+condition_size,hidden_size,latent_dims).to(device)
        self.decoder = Decoder(latent_dims+condition_size,hidden_size,output_size).to(device)

    def forward(self, x,c):
        
        z = self.encoder(x,c)
        
        return self.decoder(z,c)


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
    input_size=28*28
    output_size= 28*28
    hidden_size=512
    cvae = VariationalAutoencoder(input_size,output_size,hidden_size,latent_dims,condition_size)  # GPU
    data = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data',
                                   transform=torchvision.transforms.ToTensor(),
                                   download=True),
        batch_size=128,
        shuffle=True)
    
    cvae = train(cvae, data,epochs=20)
    plot_pre_imgs(cvae,latent_dims)
    plt.show()
    torch.save(cvae,"cvae.pth")