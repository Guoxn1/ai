from tqdm import tqdm
import torch 
import torchvision
from torch import nn
from torchvision import transforms,datasets
import torchvision.transforms as trans
import os
import sys 
import matplotlib.pyplot as plt


def data_loader(data_path,batch_size):
    transforms1 = {
        "train":transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ]),
        "test":transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
            
        ])
    }
    train_dataset = datasets.ImageFolder(root=os.path.join(data_path,"train"),transform=transforms1["train"])
    test_dataset = datasets.ImageFolder(root=os.path.join(data_path,"val"),transform=transforms1["test"])

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size,shuffle=False)
    return train_loader,test_loader

def get_net(num_classes):
    model = torchvision.models.alexnet(pretrained=True)
    model_fc = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(in_features=model_fc,out_features=num_classes)
    return model


def test(net,test_loader,device):
    net.eval()
    acc_num = torch.zeros(1).to(device)
    sample_num = 0
    test_bar = tqdm(test_loader,file=sys.stdout,ncols=100)
    with torch.no_grad(): 
        for data in test_bar:
            images,label = data
            sample_num += images.shape[0]
            images = images.to(device)
            label = label.to(device)
            output = net(images)
            pred_class = torch.max(output,dim=1)[1]
            acc_num += torch.eq(pred_class,label).sum()
    test_acc = acc_num.item()/sample_num
    return test_acc

def train(net,train_loader,loss_func,optimzer,lr,device):
    net.train()
    acc_num = torch.zeros(1).to(device)
    sample_num = 0
    train_bar = tqdm(train_loader,file=sys.stdout,ncols=100)
    for data in train_bar:
        images,label = data
        sample_num += images.shape[0]
        images = images.to(device)
        label = label.to(device)
        optimzer.zero_grad()
        output = net(images)
        pred_class = torch.max(output,dim=1)[1]
        acc_num += torch.eq(pred_class,label).sum()
        loss = loss_func(output,label)
        loss.backward()
        optimzer.step()
    train_acc = acc_num.item()/sample_num
    return train_acc

def plot_acc(epochs,train_acc_li,test_acc_li):
    plt.plot(range(1,epochs+1), train_acc_li, label="train_acc",color="red")
    plt.plot(range(1,epochs+1), test_acc_li, label="test_acc",color="blue")
    plt.xlabel("epochs")
    plt.ylabel("acc")
    plt.legend()

    plt.title("epoch-acc")
    plt.show()

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = "./data"
    batch_size = 64
    train_loader,test_loader = data_loader(data_path,64)
    num_classes = 5
    net = get_net(num_classes)
    net.to(device)
    lr = 0.0001
    epochs = 20
    loss_func = nn.CrossEntropyLoss()
    optimzer = torch.optim.Adam(net.parameters(),lr=lr)
    save_path = os.path.abspath(os.path.join(os.getcwd(),"result/alexnet"))
    if not os.path.exists(save_path):    
        os.makedirs(save_path)
    print(f"using {device}---")

    train_acc_li,test_acc_li = [],[]
    for epoch in range(epochs):
        train_acc_li.append(train(net,train_loader,loss_func,optimzer,lr,device))
        test_acc_li.append(test(net,test_loader,device)) 

    torch.save(net.state_dict(), os.path.join(save_path, "pre_AlexNet.pth") )
    print(train_acc_li[-1],test_acc_li[-1])
    plot_acc(epochs,train_acc_li,test_acc_li)
    print("train is finished---")

main()