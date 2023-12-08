import torch
from torchvision import datasets,transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from ae import Line_coder

def psnr(img1, img2):
    mse = np.mean((img1-img2)**2)
    if mse == 0:
        return float('inf')
    else:
        return 20 * np.log10(255/np.sqrt(mse))


def retrieve_images_new(query_image, train_loader, model, n=5):
    model.to(device)
    model.eval()
    query_image = query_image.view(query_image.size(0),-1).to(device)
    distances = []

    for img, _ in train_loader:
        img = img.view(img.size(0), -1).to(device)
        features = model.encoder(img)
        new_images = model.decoder(features)
        ls = []
        for images in new_images:
            resized_image = images.view([28, 28])
            raw_image = query_image.view([28, 28])
            dist = ssim(np.array(raw_image.cpu().detach().numpy()), np.array(resized_image.cpu().detach().numpy()), data_range=1)
            #dist = psnr(np.array(raw_image.cpu().detach().numpy()), np.array(resized_image.cpu().detach().numpy()))
            ls.append(dist)
        l = list(zip(ls, img.cpu().detach().numpy()))
        distances.extend(l)

    distances.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in distances[:n]]  # 返回最近的n张图片


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
        plt.title(f'Retrieved {i - 1}')
        plt.axis('off')

    plt.show()

def main():
    model = torch.load('model.pth')
        # 数据预处理：转换为Tensor并标准化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 下载并加载MNIST数据集
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    
    
    test_dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
    for img, _ in test_loader:
        query_image = img.view(img.size(0), -1)
        break  # 只取第一张图片
    # 假设 retrieve_images 和 model 已经定义
    retrieved_images = retrieve_images_new(query_image, train_loader, model, n=5)
    # 假设 visualize_retrieval 已经定义
    visualize_retrieval(query_image.cpu().squeeze(), [img.squeeze() for img in retrieved_images])

device = "cuda"
main()