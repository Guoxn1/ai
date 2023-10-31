import os
import argparse
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from load_data import iris_load

parse = argparse.ArgumentParser()
parse.add_argument("--num_classes",type=int,default=100,help="the number of classes")
parse.add_argument("--epochs",type=int,default=20,help="the number of training epoch")
parse.add_argument("--batch_size",type=int,default=16,help="batch size of training")
parse.add_argument("--lr",type=float,default=0.005,help="star learning rate")
parse.add_argument("--data_path",type=str,default="深度学习基础\神经网络\基于神经网络的鸾尾花分类\Iris_data.txt")
parse.add_argument("--device",default="cuda",help="device id(cpu)")
opt = parse.parse_args()

class Iris_network(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(Iris_network,self).__init__()
        self.layer1 = nn.Linear(in_dim,10)
        self.layer2 = nn.Linear(10,6)
        self.layer3 = nn.Linear(6,3)
    
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
    


def test(model,data):
    model.eval()
    acc_num = 0.0

    with torch.no_grad():
        for data1 in data:
            datas,label  = data1
            output = model(datas.to(device))

            predict = torch.max(output,dim=1)[1]
            acc_num += torch.eq(predict,label.to(device)).sum().item()
    accuratcy = acc_num/len(data)
    return accuratcy

def train(train_loader,validate_loader):
    best_val_accuracy = 0.0
    patience = 3
    no_improvement_epochs = 0
    lr_decay_factor = 0.98
    model = Iris_network(4,3).to(device)
    loss_function = nn.CrossEntropyLoss()
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(pg,lr=opt.lr)

    save_path = "result/weights"
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    
    for epoch in range(opt.epochs):
        model.train()

        train_bar = tqdm(train_loader, file=sys.stdout, ncols=100)

        for data in train_bar:
            datas,label = data
            label = label.squeeze(-1) 
            #data.shape[0] 表示当前小批量数据中的样本数量。
            optimizer.zero_grad()
            outputs = model(datas.to(device))
            loss = loss_function(outputs,label.to(device))
            loss.backward()
            optimizer.step()

        val_accurate = test( model, validate_loader)
        print('[epoch %d]  val_accuracy: %.3f' %  (epoch + 1, val_accurate))  
        # 在这里，可以根据验证集返回的正确率调整超参数，比如学习率

        if val_accurate > best_val_accuracy:
            torch.save(model.state_dict(), os.path.join(save_path, "NN.pth") )
            best_val_accuracy = val_accurate
        else:
            no_improvement_epochs += 1
            # 如果连续多个轮次没有改善，则降低学习率
            if no_improvement_epochs >= patience:
                # 使用学习率衰减策略，如将学习率乘以一个衰减因子
                new_learning_rate = opt.lr * lr_decay_factor
                # 更新优化器中的学习率
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_learning_rate
                # 打印学习率调整信息
                print('Learning rate adjusted to %.6f' % new_learning_rate)
                # 重置没有改善的轮次计数器
                no_improvement_epochs = 0

    return model

def get_loader(datapath):
    data = iris_load("Iris_data.txt")
    train_size = int(len(data)*0.7)
    validate_size = int(len(data)*0.2)
    test_size = int(len(data)*0.1)

    train,validate,test = torch.utils.data.random_split(data,[train_size,validate_size,test_size])

    train_loader = DataLoader(train,batch_size=opt.batch_size,shuffle=False)
    validate_loader = DataLoader(validate,batch_size=1,shuffle=False)
    test_loader = DataLoader(test,batch_size=1,shuffle=False)
    print("Training set data size:", len(train_loader)*opt.batch_size, ",Validating set data size:", len(validate_loader), ",Testing set data size:", len(test_loader)) 

    return train_loader,validate_loader,test_loader

def main(args):
    
    train_loader,validate_loader,test_loader = get_loader("深度学习基础\神经网络\基于神经网络的鸾尾花分类\Iris_data.txt")

    model = train(train_loader,validate_loader)

    test_accurancy = test(model,test_loader)
    print(' test_accuracy: %.3f' %  ( test_accurancy))  




if __name__ == '__main__': 
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    main(opt)


    

