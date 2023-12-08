import torch
from torchvision import datasets,transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


# 定义线性自编码器
class Line_coder(nn.Module):
    def __init__(self):
        super(Line_coder,self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28,256),
            nn.ReLU(True),
            nn.Linear(256,64),
            nn.ReLU(True),
            nn.Linear(64,16),
            nn.ReLU(True),
            nn.Linear(16,3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3,16),
            nn.ReLU(True),
            nn.Linear(16,64),
            nn.ReLU(True),
            nn.Linear(64,256),
            nn.ReLU(),
            nn.Linear(256,28*28),
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
            # 扁平化 成为一维向量 以便输入到线性网络中
            img = img.view(img.size(0),-1)
            img = img.to(device)
            
            output = model(img)
            loss = loss_fn(output,img)

            # 反向传播
            optim.zero_grad()
            loss.backward()
            optim.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

def retriveed_images(query_image,train_loader,model,n,device):
    
    query_image = query_image.view(query_image.size(0),-1).to(device)
    query_feature = model.encoder(query_image)
    distances = []
    for img,_ in train_loader:
        img = img.view(img.size(0),-1).to(device)
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
        query_image = img.view(img.size(0),-1).to(device)
        break
    retriveed_image = retriveed_images(query_image,train_loader,model,5,device)
    visualize_retrieval(query_image.cpu().squeeze(), [img.squeeze() for img in retriveed_image])

def main():
    device = "cuda"
    model = Line_coder()
    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(),lr=1e-3)
    epochs = 5
    train_loader,test_loader = data_load()
    train(model,loss_fn,optim,epochs,train_loader,device)
    torch.save(model,"model.pth")

    test(model,test_loader,train_loader,device)

# main()