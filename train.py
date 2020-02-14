import time
import sys
import os
import numpy as np
from sklearn.metrics import accuracy_score as acc
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torchvision.models as models

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim

from Net import double 
from Mydataset import MyDataSet, ValDataSet
WANDB = 1
if WANDB == True:
    import wandb
    wandb.init(project = "mnist_add")

def main():

    epochs = 20
    batch_size =64
    device_id= [0]


    train_transform = transforms.Compose(
        [    # 360度ランダムで画像を回転する
            # transforms.Resize((32,32)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5,))])

    test_transform = transforms.Compose(
             [
            # transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])
    train_set = MyDataSet(train_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,drop_last=True)

    test_set = ValDataSet(test_transform)
    val_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    net = double()
    net = net.to(device)

    #criterion = nn.MSELoss()
    #criterion = nn.SmoothL1Loss()
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr = 0.001)
    # optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum=0.9)
    #optimizer = optim.SGD(net.parameters(), lr = 0.008, momentum=0.9, weight_decay = 0.008)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    train_log = []
    test_log=[]
    test_acc=[]

    print("start training")

    for epoch in range(epochs):

        t1 = time.time()
        total_train_loss = [0.0,0]
        total_test_loss = [0.0,0]

        correct_train = 0
        total_train = 0
        for i, data in enumerate(train_loader, 0):
            #get input
            img1, img2, labels = data
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(img1, img2)
            train_loss = criterion(outputs, labels)

            train_loss.backward()
            optimizer.step()

            total_train_loss[0] += train_loss.item()
            total_train_loss[1] += 1
            _, predicted = torch.max(outputs.data, 1)

            total_train += labels.size(0)
            correct_train += (predicted == labels).cpu().sum().item()

            sys.stdout.write('\r[Epoch %d/%d] [Batch %d/%d] [Train Loss: %.4f]' % (epoch, epochs, i, len(train_loader), total_train_loss[0]/total_train_loss[1]))

        scheduler.step()
        t2 = time.time()
        correct_test = 0
        total_test = 0

        result = [[0 for i1 in range(10)] for i2 in range(10)]
        result_num = [[0 for i1 in range(10)] for i2 in range(10)]
        with torch.no_grad():
             for data in val_loader:
                img1, img2, labels = data
                img1 = img1.to(device)
                img2 = img2.to(device)
                num1,num2,labels = labels[:,0], labels[:,1], labels[:,2]
                labels = labels.to(device)
                outputs = net(img1, img2)

                test_loss = criterion(outputs, labels)
                total_test_loss[0] += test_loss.item()
                total_test_loss[1] += 1

                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).cpu().sum().item()

                ans = (predicted == labels).cpu().numpy()
                for i in range(len(ans)):
                    n1 = num1[i]
                    n2 = num2[i]
                    result[n1][n2] += ans[i]
                    result_num[n1][n2] += 1


        t3 = time.time()
        print("train_acc :", correct_train, " / " ,total_train, "",correct_train/total_train*100,"%")
        print("test_acc  :", correct_test, " / " ,total_test, "",correct_test/total_test*100,"%")
        for i in range(10):
            for j in range(10):
                print(i,j,"{}/{},{:.1f}%".format(result[i][j],result_num[i][j],result[i][j]/result_num[i][j]*100))

        if WANDB == True:
            wandb_log = {}
            wandb_log["epoch"] = epoch
            wandb_log["train_acc"] = correct_train/total_train*100
            wandb_log["test_acc"] = correct_test/total_test*100
            wandb.log(wandb_log)

    net = net.cpu()
    torch.save(net.state_dict(),"weight/classificate.pth")

    print("finish")


if __name__=="__main__":
    main()
