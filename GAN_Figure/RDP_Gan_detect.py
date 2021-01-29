import numpy as np
import matplotlib
import sys
import pylab
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import os
import copy
import time

batch_size = 1
num_epochs = 10000
z_dimension = 100

digits = [0]
# Noise scale
set_sigma = [0] 
use_gpu = torch.cuda.is_available()
# Image processing
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# MNIST dataset
mnist = datasets.MNIST(
    root='../data/', train=True, transform=img_transform, download=True)
test_dataset = datasets.MNIST(
    root='../data/', train=False, transform=img_transform)

def unique_index(L,f):
    return [i for (i,value) in enumerate(L) if value==f]

def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 28, 28)
    return out

def get_noise(sigma):
    Noise=np.random.normal(0,sigma)
    return Noise 

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):

        image, label = self.dataset[int(self.idxs[item])]
        return image, label

# Discriminator
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2))
        self.layer2 = nn.Sequential(        
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2))
        self.layer3 = nn.Sequential(        
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2))
        self.layer4 = nn.Sequential(        
            nn.Linear(256, 1), 
            nn.Sigmoid())

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer2(x2)
        x = self.layer4(x3) 
        return x

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

criterion_test = nn.CrossEntropyLoss()
neural_net = neuralNetwork(28 * 28, 256, 256, 10)
neural_net.load_state_dict(torch.load('../classifier_Mnist/' + 'classifier_Mnist.pth'))
if torch.cuda.is_available():
    neural_net = neural_net.cuda()

for k in set_sigma:
    sigma = copy.deepcopy(k)

    #if not os.path.exists('./img_noise_loss_{}'.format(sigma)):
    #    os.mkdir('./img_noise_loss_{}'.format(sigma))
    if not os.path.exists('./loss_std{}'.format(sigma)):    
        os.mkdir('./loss_std{}'.format(sigma))
        
    for digit in digits:
        D = discriminator()
        G = generator()
        if torch.cuda.is_available():
            D = D.cuda()
            G = G.cuda()
        # Binary cross entropy loss and optimizer
        criterion = nn.BCELoss()
        d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
        g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)
        
        # Data loader
        num_labels = mnist.train_labels.numpy()
        classes = np.unique(num_labels)
        classes_index = []
        digit_size = 60000
        for m in range(len(classes)):
            classes_index.append(unique_index(num_labels, classes[m]))
            if len(classes_index[m])< digit_size:
                digit_size = len(classes_index[m])
                group_size = int(np.floor(digit_size/batch_size))
        for m in range(len(classes)):
            # classes_index[m][group_size*batch_size:]=[]
            classes_index[m][1:]=[]
        dataloader = torch.utils.data.DataLoader(
            dataset = DatasetSplit(mnist, classes_index[digit]), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Start training
        record_matrix_list = []
        real_loss, real_acc, fake_loss, fake_acc = [], [], [], []
        g_loss_list, d_loss_list = [], []
        for epoch in range(num_epochs):
            eval_loss_fake, eval_loss_real = 0., 0.
            eval_acc_fake, eval_acc_real = 0. ,0.
            eval_acc_gan = 0.
            
            for i, (img, label) in enumerate(dataloader):
                num_img = img.size(0)
                # =================train discriminator
                img = img.view(num_img, -1)
                real_img = Variable(img).cuda()
                real_label = Variable(torch.ones(num_img)).cuda()
                fake_label = Variable(torch.zeros(num_img)).cuda()
        
                # compute loss of real_img
                real_out = D(real_img)
                        
                d_loss_real = criterion(real_out, real_label)
                
                #d_loss_real=d_loss_real+noise
                real_scores = real_out  # closer to 1 means better
        
                # compute loss of fake_img
                z = Variable(torch.randn(num_img, z_dimension)).cuda()
                fake_img, g_layer2, g_layer1 = G(z)
                fake_out = D(fake_img)
                d_loss_fake = criterion(fake_out, fake_label)
                fake_scores = fake_out  # closer to 0 means better
        
                # bp and optimize
                d_loss = d_loss_real + d_loss_fake
                d_optimizer.zero_grad()
                d_loss.backward()
            
                d_optimizer.step()
        
                # ===============train generator
                # compute loss of fake_img
                z = Variable(torch.randn(num_img, z_dimension)).cuda()
                fake_img, _, _ = G(z)
                output = D(fake_img)
                noise=get_noise(sigma)
                g_loss = criterion(output, real_label)+noise
        
                # bp and optimize
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()
        
                if (i + 1) % 5 == 0:
                    print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} '
                          'D real: {:.6f}, D fake: {:.6f}'.format(
                              epoch, num_epochs, d_loss.item(), g_loss.item(),
                              real_scores.data.mean(), fake_scores.data.mean()))
         
                if use_gpu:
                    label = label.cuda()
                out_fake = neural_net(fake_img)
                out_real = neural_net(real_img)
                loss_fake = criterion_test(out_fake, label)
                loss_real = criterion_test(out_real, label)
                eval_loss_fake += loss_fake.item()
                eval_loss_real += loss_real.item()
                _, pred_fake = torch.max(out_fake, 1)
                _, pred_real = torch.max(out_real, 1)
                eval_acc_fake += (pred_fake == label).float().mean()
                eval_acc_real += (pred_real == label).float().mean() 
                
                fake_label_D = Variable(torch.zeros(num_img)).cuda()
                for j in range(len(fake_out)):
                    if float(fake_out[j].cpu())>= 0.5:
                        fake_label_D[j] = 1            
                eval_acc_gan += (fake_label_D == fake_label).float().mean() 
            # record_matrix_list.append(record_matrix)
            g_loss_list.append(g_loss.item())
            d_loss_list.append(d_loss.item())
            real_loss.append(eval_loss_real/(i + 1))
            real_acc.append(float(eval_acc_real/(i + 1)))
            fake_loss.append(eval_loss_fake/(i + 1))
            fake_acc.append(float(eval_acc_fake/(i + 1)))        
            print(f'\nGan Acc: {eval_acc_gan/(i + 1):.6f}\n')            
            print(f'Real Img Loss: {eval_loss_real/(i + 1):.6f}, Real Img Acc: {eval_acc_real/(i + 1):.6f}\n')        
            print(f'Fake Img Loss: {eval_loss_fake/(i + 1):.6f}, Fake Img Acc: {eval_acc_fake/(i + 1):.6f}\n')            
            print(f'Gan Loss: {g_loss:.6f}, Dis loss: {d_loss:.6f}\n')
                         
            if epoch == 0:     
                real_images = to_img(real_img.cpu().data)
                save_image(real_images, './loss_std{}/record-{}.png'.format(sigma,digit, epoch + 1))
        
            if (epoch+1)%50 == 0:  
                fake_images = to_img(fake_img.cpu().data)
                save_image(fake_images, './loss_std{}/fake_images_digit-{}-{}.png'.format(sigma,digit, epoch + 1))
      
        with open('./loss_std{}/real_images_digit-{}.txt'.format(sigma,digit),'w',encoding='utf-8') as f:
            f.write('real_loss:\n')
            f.write(str(real_loss))
            f.write('\nreal_acc:\n')
            f.write(str(real_acc))
            f.write('\nfake_loss:\n')
            f.write(str(fake_loss))
            f.write('\nfake_acc:\n')
            f.write(str(fake_acc)) 
            f.write('\ng_loss:\n')
            f.write(str(g_loss_list))
            f.write('\nd_loss:\n')
            f.write(str(d_loss_list))
        plt.figure()
        plt.plot(range(len(real_acc)), real_acc, label='Real acc')
        plt.plot(range(len(fake_acc)), fake_acc, label='Fake acc')
        plt.ylabel('train_acc')
        plt.xlabel('num_epoches')
        plt.grid(linestyle = "--")    
        plt.savefig('./loss_std{}/detect_result_std{}_digit-{}.pdf'.format(sigma, sigma, digit)) 
        
        plt.figure()
        plt.plot(range(len(g_loss_list)), g_loss_list, label='G loss')
        plt.plot(range(len(d_loss_list)), d_loss_list, label='D loss')
        plt.ylabel('train_loss')
        plt.xlabel('num_epoches')
        plt.grid(linestyle = "--")    
        plt.savefig('./loss_std{}/train_loss_std{}_digit-{}.pdf'.format(sigma, sigma, digit))                      
          
        timeslot = int(time.time())
        dt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timeslot))
        
        torch.save(G.state_dict(), './loss_std{}/generator_digit-{}.pth'.format(sigma, digit))
        torch.save(D.state_dict(), './loss_std{}/discriminator_digit-{}.pth'.format(sigma, digit))
