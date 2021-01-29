"""
@author:  KANGWEI
"""

import torch
import os
import time
import csv
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer

# MLP
class neuralNetwork(nn.Module):
    def __init__(self, in_dim, n_hidden_1, out_dim):
        super(neuralNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.ReLU(True))
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, out_dim),
            nn.ReLU(True))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
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
    
batch_size = 16
learning_rate = 0.005
num_epochs = 20
use_gpu = torch.cuda.is_available()
dataset = 'Adult' # 'Adult' or 'Mnist' or 'FashionMnist'
style = 'test' # 'train' or 'test'
set_sigma = [0.0]
dp_mechanism = 'loss' # 'loss' or 'para'
# set_GAN_epochs = [602]
set_GAN_epochs = range(1502,15003,1500)
encode = False

if not os.path.exists('./classifier_{}'.format(dataset)):
    os.mkdir('./classifier_{}'.format(dataset))
    
if style == 'test':
    ########-Test for ADULT-#########
    for sigma in set_sigma:
        detect_acc, detect_loss = [], []
        for GAN_epochs in set_GAN_epochs:
            
            print('*' * 15,f'STD: {sigma}, GAN epochs: {GAN_epochs}','*' * 15)
            ###-data location-###
            csv_path = ('C:/Users/hp/Desktop/dp_gan/tableGAN-master/samples/Adult/')
            #fake_csv_path = (csv_path+'encode_Agg_data.csv'.format(dp_mechanism, sigma, GAN_epochs))
            fake_csv_path = (csv_path+'Adult_{}_std{}_epoch{}_fake.csv'.format(dp_mechanism, sigma, GAN_epochs))
                
            with open(fake_csv_path,'r',encoding='utf8')as fp:
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
            model.load_state_dict(torch.load('./classifier_{}/'.format(dataset)+'classifier_Adult.pth'))
            #model.load_state_dict(torch.load('C:/Users/hp/Desktop/dp_gan/classifier_Adult/1234.pth'))  
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
#                eval_loss += loss.item()
#                _, pred = torch.max(out, 1)
#                eval_err += (pred == label).float().mean()
#                eval_acc = float(eval_err.cpu()/(i+1))
                eval_loss += loss.item()
                _, pred = torch.max(out, 1)
                eval_acc += (pred == label).float().mean()
                #print('pred:', pred,eval_acc)
            detect_loss.append(eval_loss/len(test_loader))
            detect_acc.append(eval_acc)
            #print('pred:', pred)
            print(f'Test Loss: {eval_loss/len(test_loader):.6f}, Acc: {eval_acc/len(test_loader):.6f}\n')
            with open(csv_path+'Adult_{}_std{}_detect.txt'\
                      .format(dp_mechanism, sigma),'w',encoding='utf-8') as f:
                f.write('detect_loss:\n')
                f.write(str(detect_loss))
                f.write('\ndetect_acc:\n')
                f.write(str(detect_acc))

########-Train for different datasets-#########
elif style == 'train':

    if dataset == 'Mnist':
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        train_dataset = datasets.MNIST(
            root='./data/', train=True, transform=img_transform, download=False)
        
        test_dataset = datasets.MNIST(
            root='./data/', train=False, transform=img_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        model = neuralNetwork(28 * 28, 128, 10)
        # model.load_state_dict(torch.load('./classifier_Mnist/' + 'classifier_Mnist.pth'))
        if use_gpu:
            model = model.cuda()
            
        criterion = nn.CrossEntropyLoss()
    

    if dataset == 'FashionMnist':
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
         
        train_dataset = datasets.FashionMNIST(
            root='./data/', train=True, transform=img_transform, download=False)
        
        test_dataset = datasets.FashionMNIST(
            root='./data/', train=False, transform=img_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        if style == 'train':
            model = neuralNetwork(28 * 28, 512, 256, 10)
            if use_gpu:
                model = model.cuda()
        criterion = nn.CrossEntropyLoss()
        
    elif dataset == 'Adult':
        
        if encode:
            fpath = './data/'+'Adult.dat'
            X, y = load_dat(fpath, minmax=(0, 1), normalize=False, bias_term=True)
            num_train = 40000
            
            train_dataset = []
            for i in range(len(y)):
                train_indiv = Variable(torch.FloatTensor(np.expand_dims(X[i], axis = 0)))
                train_indiv = [train_indiv, int(y[i])]  
                train_dataset.append(train_indiv)
            train_loader = DataLoader(train_dataset[:num_train], batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(train_dataset[num_train:], batch_size=batch_size, shuffle=False)
            
            # model = net_Binaryclassify(124, 256, 64, 2)
            model = logsticRegression(124, 2)
            if use_gpu:
                model = model.cuda()
            # criterion = nn.MSELoss()
            criterion = nn.CrossEntropyLoss()
        
        else:
        
            # Read Tabular data
            csv_path = './data/Adult/'+'encode_Agg_data.csv'
            #csv_path = './data/Tabular_data/'+'Adultall.csv'
            fake_csv_path = './data/Adult/'+'encode_Agg_data.csv'
            with open(csv_path,'r',encoding='utf8')as fp:
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
        
            with open(fake_csv_path,'r',encoding='utf8')as fp:
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
         
            model = logsticRegression(35, 2)
            # model = net_Binaryclassify(11, 16, 8, 1)
            if use_gpu:
                model = model.cuda()
            # criterion = nn.BCELoss()
            criterion = nn.CrossEntropyLoss()
    
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        print('*' * 15,f'Epoch: {epoch+1}','*' * 15)
        #print(f'epoch {epoch+1}')
        running_loss = 0.0
        running_acc = 0.0
        for i, data in enumerate(train_loader, 1):
            img, label = data
            img = img.view(img.size(0), -1)
    
            if use_gpu:
                img = img.cuda()
                label = label.cuda()
            # 向前传播
            out = model(img)
            loss = criterion(out, label)
            running_loss += loss.item()
            # print('loss:', loss, running_loss)
            _, pred = torch.max(out, 1)
            # print('Lab result:', label)
            running_acc += (pred == label).float().mean()
            # 向后传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            if i % 50 == 0:
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
            print('Pre result:', pred)
        print(f'Test Loss: {eval_loss/len(test_loader):.6f}, Acc: {eval_acc/len(test_loader):.6f}\n')
    
    timeslot = int(time.time())
    dt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timeslot))
    #torch.save(model.state_dict(),'./classifier_{}/classifier_{}.pth'.format(dataset, timeslot))
    torch.save(model.state_dict(),'./classifier_{}/1234.pth'.format(dataset))