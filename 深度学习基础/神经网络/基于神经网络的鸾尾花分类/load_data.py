from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

class iris_load(Dataset):
    def __init__(self,datapath:str,transform=None):
        self.datapath = datapath
        self.transform = transform
        print(datapath)
        #assert os.path.exists(datapath),"dataset doesnt exist"

        df = pd.read_csv(self.datapath,names=[0,1,2,3,4])
        # 把标签转换为数字
        d = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
        df[4] = df[4].map(d)

        data = df.iloc[:,0:4]
        label = df.iloc[:,4]

        data = np.array(data)
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        label = np.array(label)

        self.data = torch.from_numpy((np.array(data,dtype='float32')))
        self.label= torch.from_numpy(np.array(label,dtype='int64') ) 
        self.data_num = len(label)

    def __len__(self):
        return self.data_num
    
    def __getitem__(self, idx):
        self.data = list(self.data)
        self.label = list(self.label)
        return self.data[idx], self.label[idx]

# data = iris_load("深度学习基础\神经网络\基于神经网络的鸾尾花分类\Iris_data.txt")
# print(data)
