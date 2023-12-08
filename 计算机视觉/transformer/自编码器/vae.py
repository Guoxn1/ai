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

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
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

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, 28, 28))


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_size,output_size,hidden_size,latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(input_size,hidden_size,latent_dims).to(device)
        self.decoder = Decoder(latent_dims,hidden_size,output_size).to(device)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def train(autoencoder, data, epochs=10):
    device="cuda"
    autoencoder = autoencoder.to(device)
    opt = torch.optim.Adam(autoencoder.parameters())
    for epoch in range(epochs):
        for x, _ in data:
            x = x.to(device)  # GPU
            x_hat = autoencoder(x)
            loss = ((x - x_hat) ** 2).sum() + autoencoder.encoder.kl
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(epoch)
    return autoencoder




def plot_reconstructed(autoencoder, r0=(-5, 10), r1=(-10, 5), n=12):
    w = 28
    img = np.zeros((n * w, n * w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = autoencoder.decoder(z)
            x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
            img[(n - 1 - i) * w:(n - 1 - i + 1) * w, j * w:(j + 1) * w] = x_hat
    plt.imshow(img, extent=[*r0, *r1])


def interpolate(autoencoder, x_1, x_2, n=12):
    z_1 = autoencoder.encoder(x_1)
    z_2 = autoencoder.encoder(x_2)
    z = torch.stack([z_1 + (z_2 - z_1) * t for t in np.linspace(0, 1, n)])
    interpolate_list = autoencoder.decoder(z)
    interpolate_list = interpolate_list.to('cpu').detach().numpy()

    w = 28
    img = np.zeros((w, n * w))
    for i, x_hat in enumerate(interpolate_list):
        img[:, i * w:(i + 1) * w] = x_hat.reshape(28, 28)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])

def plot_pre_img(vae,x_1):
    z1 = vae.encoder(x_1)
    interpolate_list = vae.decoder(z1)
    interpolate_list = interpolate_list.to('cpu').detach().numpy()
    interpolate_list = interpolate_list.squeeze()
    plt.imshow(interpolate_list)

if __name__ == "__main__":
    # device="cpu"
    latent_dims = 2
    input_size=output_size=28*28
    hidden_size=512
    vae = VariationalAutoencoder(input_size,output_size,hidden_size,latent_dims)  # GPU
    data = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data',
                                   transform=torchvision.transforms.ToTensor(),
                                   download=True),
        batch_size=128,
        shuffle=True)

    vae = train(vae, data,epochs=5)
    plot_reconstructed(vae, r0=(-3, 3), r1=(-3, 3))
    plt.show()
    x, y = next(iter(data)) # hack to grab a batch
    x_1 = x[y == 2][1].to(device)  # find a 1
    x_2 = x[y == 0][1].to(device)  # find a 0

    interpolate(vae, x_1, x_2, n=20)
    plt.show()
    plot_pre_img(vae, x_1)
    plt.show()
    torch.save(vae,"vae.pth")