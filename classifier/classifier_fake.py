# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 21:37:26 2020

@author: WEIKANG
"""

import torch
import os
import time
import csv
import copy
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer

def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 28, 28)
    return out

# Generator
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(True))
        self.layer2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(True))
        self.layer3 = nn.Sequential(
            nn.Linear(256, 784),
            nn.Tanh())

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x = self.layer3(x2)        
        return x, x2, x1

# MLP
class neuralNetwork(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(neuralNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.ReLU(True))
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.ReLU(True))
        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden_2, out_dim),
            nn.ReLU(True))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
    
class net_Binaryclassify(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(net_Binaryclassify, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.LeakyReLU(0.2))
        self.layer2 = nn.Sequential(        
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.LeakyReLU(0.2))
        self.layer3 = nn.Sequential(        
            nn.Linear(n_hidden_2, out_dim), 
            nn.Sigmoid())

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x = self.layer3(x2) 
        return x
    
class logsticRegression(nn.Module):
    def __init__(self, in_dim, n_class):
        super(logsticRegression, self).__init__()
        self.logstic = nn.Sequential(
            nn.Linear(in_dim, n_class), 
            nn.Sigmoid())

    def forward(self, x):
        out = self.logstic(x)
        return out
    
def load_dat(filepath, minmax=None, normalize=False, bias_term=True):   
    ###-load a dat file-###
    lines = np.loadtxt(filepath)
    labels = lines[:, -1]
    features = lines[:, :-1]

    N, dim = features.shape

    if minmax is not None:
        minmax = MinMaxScaler(feature_range=minmax, copy=False)
        minmax.fit_transform(features)

    if normalize:
        # make sure each entry's L2 norm is 1
        normalizer = Normalizer(copy=False)
        normalizer.fit_transform(features)

    if bias_term:
        X = np.hstack([np.ones(shape=(N, 1)), features])
    else:
        X = features

    return X, labels
    
batch_size = 128
learning_rate = 0.005
num_epochs = 100
z_dimension = 100
use_gpu = torch.cuda.is_available()
set_sigma = [1.0, 5.0, 10, 50]
dataset = 'Adult'
set_dp_mechanism = ['para','loss'] # 'loss' or 'para'
GAN_epochs = 3002
encode = False

num_experiments = 5


final_train_loss = [[0 for i in range(len(set_sigma))] for j in range(len(set_dp_mechanism))]
final_train_acc = [[0 for i in range(len(set_sigma))] for j in range(len(set_dp_mechanism))]
final_test_loss = [[0 for i in range(len(set_sigma))] for j in range(len(set_dp_mechanism))]
final_test_acc = [[0 for i in range(len(set_sigma))] for j in range(len(set_dp_mechanism))]
for s in range(len(set_sigma)):
    sigma = copy.deepcopy(set_sigma[s])
    for t in range(len(set_dp_mechanism)):
        dp_mechanism = copy.deepcopy(set_dp_mechanism[t])
        print('*' * 15,f'STD: {sigma}, DP mechanism: {dp_mechanism}','*' * 15)
        if dataset == 'Adult':
            
            ###-data loader-###
            path = ('./data/FakeAdult/')
            real_csv_path = (path + 'encode_Adult.csv')
            if not os.path.exists(path+'Adult_{}_std{}_epoch{}_fake.csv'\
                                  .format(dp_mechanism, sigma, GAN_epochs)):
                continue
            else:
                fake_csv_path = (path+'Adult_{}_std{}_epoch{}_fake.csv'.format(dp_mechanism, sigma, GAN_epochs))
            
            with open(fake_csv_path,'r',encoding='utf8')as fp:
                train_data = [i for i in csv.reader(fp)]
            train_dataset = []
            for i in range(1, len(train_data)):
                train_indiv = []
                for j in range(len(train_data[1])-1):
                    train_indiv.append(float(train_data[i][j]))
                train_indiv = Variable(torch.FloatTensor(np.expand_dims(train_indiv, axis = 0)))
                train_indiv = [train_indiv, int(float(train_data[i][j+1]))]  
                train_dataset.append(train_indiv)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            # train_loader = Variable(torch.FloatTensor(train_loader)).cuda()
        
            with open(real_csv_path,'r',encoding='utf8')as fp:
                test_data = [i for i in csv.reader(fp)]
            test_dataset = []
            for i in range(1, len(test_data)):
                test_indiv = []
                for j in range(len(test_data[1])-1):
                    test_indiv.append(float(test_data[i][j]))
                test_indiv = Variable(torch.FloatTensor(np.expand_dims(test_indiv, axis = 0)))
                test_indiv = [test_indiv, int(float(test_data[i][j+1]))]
                test_dataset.append(test_indiv)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
         
            model = logsticRegression(41, 2)
            # model = net_Binaryclassify(11, 16, 8, 1)
            if use_gpu:
                model = model.cuda()
            # criterion = nn.BCELoss()
            criterion = nn.CrossEntropyLoss()
            
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)
                    
        elif dataset == 'Mnist':
            
            path = ('./data/FakeMnist/')
            
            if not os.path.exists(path+'{}_noise_{}'.format(dp_mechanism, sigma)):
                continue
            
            img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])
            
            G = generator()
            if torch.cuda.is_available():
                G = G.cuda()
        
            num_imgs = 1000
            
            train_dataset = []
            for j in range(10):
                G.load_state_dict(torch.load('./data/FakeMnist/{}_noise_{}/'.format(dp_mechanism, sigma)\
                                             +'generator_digit-{}.pth'.format(j)))
                for k in range(num_imgs):
                    z = Variable(torch.randn(1, z_dimension)).cuda()
                    fake_img, _, _ = G(z)
                    fake_imgs = to_img(fake_img)
                    fake_images = [fake_imgs[0], j]
                    train_dataset.append(fake_images)
        
        #    test_dataset = datasets.MNIST(
        #        root='./data/', train=True, transform=img_transform, download=False)
            test_dataset = datasets.MNIST(
                root='./data/', train=False, transform=img_transform)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            criterion = nn.CrossEntropyLoss()
            model = neuralNetwork(28 * 28, 256, 256, 10)
            if torch.cuda.is_available():
                model = model.cuda()
                
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)
            # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            
        test_loss, test_acc = [], []
        train_loss, train_acc = [], []
        for exper in  range(num_experiments):

            for epoch in range(num_epochs):
                
                running_loss, running_acc = 0.0, 0.0
                for i, data in enumerate(train_loader, 1):
                    img, label = data
                    img = img.view(img.size(0), -1)
            
                    if use_gpu:
                        img = img.cuda()
                        label = label.cuda()
                    out = model(img)
                    loss = criterion(out, label)
                    running_loss += loss.item()
                    # print('loss:', loss, running_loss)
                    _, pred = torch.max(out, 1)
                    # print('Lab result:', label)
                    running_acc += (pred == label).float().mean()
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()
            
                    if i % 5 == 0:
                        print(f'[{epoch+1}/{num_epochs}] Loss: {running_loss/(i+1):.6f}, Acc: {running_acc/(i+1):.6f}')
                print(f'Finish {epoch+1} epoch, Loss: {running_loss/(i+1):.6f}, Acc: {running_acc/(i+1):.6f}')
                
                model.eval()
                eval_loss = 0.
                eval_acc = 0.
                for data in test_loader:
                    img, label = data
                    img = img.view(img.size(0), -1)
                    if use_gpu:
                        img = img.cuda()
                        label = label.cuda()
                    with torch.no_grad():
                        out = model(img)
                        loss = criterion(out, label)
                    eval_loss += loss.item()
                    _, pred = torch.max(out, 1)
                    eval_acc += (pred == label).float().mean()
                # print('Pre result:', pred)
                print(f'Test Loss: {eval_loss/len(test_loader):.6f}, Acc: {eval_acc/len(test_loader):.6f}\n')
                
            train_loss.append(running_loss/len(train_loader))
            train_acc.append(float(running_acc.cpu())/len(train_loader))
            test_loss.append(eval_loss/len(test_loader))
            test_acc.append(float(eval_acc.cpu()/len(test_loader)))
            
        final_train_loss[t][s] = copy.deepcopy(sum(train_loss)/(exper+1))
        final_train_acc[t][s] = copy.deepcopy(sum(train_acc)/(exper+1))
        final_test_loss[t][s] = copy.deepcopy(sum(test_loss)/(exper+1))
        final_test_acc[t][s] = copy.deepcopy(sum(test_acc)/(exper+1))
#        with open(path+'{}_{}_std{}_detect.txt'\
#                  .format(dataset, dp_mechanism, sigma),'w',encoding='utf-8') as f:
#            f.write('train loss:\n')
#            f.write(str(train_loss))
#            f.write('\ntrain acc:\n')
#            f.write(str(train_acc))
#            f.write('\ntest loss:\n')
#            f.write(str(test_loss))
#            f.write('\ntest acc:\n')
#            f.write(str(test_acc))

data_train_acc = pd.DataFrame(index = set_dp_mechanism, columns = set_sigma, data = final_train_acc)
data_train_acc.to_csv('./data/FakeTrain/'+'train_acc_{}.csv'.format(dataset))
data_train_loss = pd.DataFrame(index = set_dp_mechanism, columns = set_sigma, data = final_train_loss)
data_train_loss.to_csv('./data/FakeTrain/'+'train_loss_{}.csv'.format(dataset))
data_test_acc = pd.DataFrame(index = set_dp_mechanism, columns = set_sigma, data = final_test_acc)
data_test_acc.to_csv('./data/FakeTrain/'+'test_acc_{}.csv'.format(dataset))
data_test_loss = pd.DataFrame(index = set_dp_mechanism, columns = set_sigma, data = final_test_loss)
data_test_loss.to_csv('./data/FakeTrain/'+'test_loss_{}.csv'.format(dataset))