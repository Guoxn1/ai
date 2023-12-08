import torch
from torchvision import datasets,transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


# 定义线性自编码器
class conv_coder(nn.Module):
    def __init__(self,laten_size):
        super(conv_coder,self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1,8,kernel_size=3,padding=1),  # [1,28,28] -> [8,28,28] 
            nn.MaxPool2d(kernel_size=2,stride=2),    # [8,28,28] -> [8,14,14]
            nn.ReLU(),
            nn.Conv2d(8,16,kernel_size=3,padding=1), #[8,14,14] -> [16,14,14]
            nn.MaxPool2d(kernel_size=2,stride=2),     #[16,14,14] -> [16,7,7]
            nn.ReLU(),
            nn.Conv2d(16,4,kernel_size=3,padding=1),  #[16,7,7]  -> [4,7,7]
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4*7*7,32),
            nn.ReLU(),
            nn.Linear(32,laten_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(laten_size,32),
            nn.ReLU(),
            nn.Linear(32,4*7*7),
            nn.ReLU(),
            nn.Unflatten(1,(4,7,7)),
            nn.ReLU(),
            nn.ConvTranspose2d(4,16,kernel_size=3,padding=1),
            nn.ReLU(),
            # 最大池化
            nn.ConvTranspose2d(16,8,kernel_size=3,padding=1,stride=2,output_padding=1),
            nn.ReLU(),
            # 最大池化
            nn.ConvTranspose2d(8,1,kernel_size=3,padding=1,stride=2,output_padding=1),
            nn.Tanh()
        )

    def forward(self,X):
        X = self.encoder(X)
        X = self.decoder(X)

        return X
    

def data_load():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])

    train_dataset = datasets.MNIST(root="../data",train=True,download=True,transform=transform)
    test_dataset = datasets.MNIST(root="../data",download=True,train=False,transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=1,shuffle=True)

    return train_loader,test_loader


def train(model,loss_fn,optim,epochs,train_loader,device):
    model.train()
    model.to(device)
    for epoch in range(epochs):
        for img,_ in train_loader:
            img = img.to(device)
            
            output = model(img)
            
            loss = loss_fn(output,img)

            # 反向传播
            optim.zero_grad()
            loss.backward()
            optim.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

def retriveed_images(query_image,train_loader,model,n,device):
    
    query_image = query_image.to(device)
    query_feature = model.encoder(query_image)
    distances = []
    for img,_ in train_loader:
        img = img.to(device)
        features = model.encoder(img)
        dist = torch.norm(features-query_feature,dim=1)
        distances.extend(list(zip(dist.cpu().detach().numpy(),img.cpu().detach().numpy())))

    distances.sort(key=lambda x:x[0])
    return [x[1] for x in distances[:n]]


def visualize_retrieval(query_image, retrieved_images):
    plt.figure(figsize=(10, 2))

    # 显示查询图片
    plt.subplot(1, len(retrieved_images) + 1, 1)
    plt.imshow(query_image.reshape(28, 28), cmap='gray')
    plt.title('Query Image')
    plt.axis('off')

    # 显示检索到的图片
    for i, img in enumerate(retrieved_images, 2):
        plt.subplot(1, len(retrieved_images) + 1, i)
        plt.imshow(img.reshape(28, 28), cmap='gray')
        plt.title(f'Retrieved {i-1}')
        plt.axis('off')

    plt.show()

def test(model,test_loader,train_loader,device):
    model.eval()
    model.to(device)
    for img,_ in test_loader:
        query_image = img.to(device)
        break
    retriveed_image = retriveed_images(query_image,train_loader,model,5,device)
    visualize_retrieval(query_image.cpu().squeeze(), [img.squeeze() for img in retriveed_image])

def main():
    device = "cuda"
    laten_size = 3
    model = conv_coder(laten_size)
    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(),lr=1e-3)
    epochs = 5
    train_loader,test_loader = data_load()
    train(model,loss_fn,optim,epochs,train_loader,device)
    torch.save(model,"model_cnn.pth")

    test(model,test_loader,train_loader,device)

main()