import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset
import torch
import csv
import os
import pickle
class MyDataSet(Dataset):

    def __init__(self,transform=None ):
        self.transform = transform
        self.data = []
        f = open("train_pair.txt","rb")
        self.data = pickle.load(f) 
        # print(self.data[0])
        # print(len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pic1, pic2, label = self.data[idx]
        data_path1 = "data/MNIST/train/"+pic1
        img1 = self.transform(Image.open(data_path1))
        data_path2 = "data/MNIST/train/"+pic2
        img2 = self.transform(Image.open(data_path2))
        #w,h = img.shape
        
        #print(data.shape,label)
        return img1,img2, torch.tensor(int(label[2]), dtype = torch.long)
        #return img1,img2, torch.tensor(label[2], dtype = torch.float32)
        #return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

class ValDataSet(Dataset):

    def __init__(self,transform=None ):
        self.transform = transform
        self.data = []
        f = open("test_pair.txt","rb")
        self.data = pickle.load(f) 
        # print(self.data[0])
        # print(len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pic1, pic2, label = self.data[idx]
        data_path1 = "data/MNIST/test/"+pic1
        img1 = self.transform(Image.open(data_path1))
        data_path2 = "data/MNIST/test/"+pic2
        img2 = self.transform(Image.open(data_path2))
        #w,h = img.shape
        
        #print(data.shape,label)
        return img1,img2, torch.tensor(label[2])
        #return img1,img2, torch.tensor(label[2], dtype = torch.float32)
        #return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


if __name__=="__main__":
    test_transform = transforms.Compose(
           [transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5, ))])
    MyDataSet(test_transform)
    val_loader = torch.utils.data.DataLoader(MyDataSet(test_transform), batch_size=16, shuffle=False)
    for i in val_loader:
        img1, img2 , label = i
        print(label)
        break
