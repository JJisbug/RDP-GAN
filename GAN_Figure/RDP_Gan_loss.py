import numpy as np
import time
import matplotlib
import sys
import pylab
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import os
import copy


def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 28, 28)
    return out

def get_noise(sigma):
    Noise=np.random.normal(0,sigma)
    return Noise 


# Discriminator
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2), nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.dis(x)
        return x


# Generator
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(True),
            nn.Linear(256, 256), nn.ReLU(True), nn.Linear(256, 784), nn.Tanh())

    def forward(self, x):
        x = self.gen(x)
        return x

batch_size = 128
num_epoch = 2000
z_dimension = 100

C1 = 1
# Noise scale
set_sigma = [0.01,0.02,0.05,0.1,0.2] 

# Image processing
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# MNIST dataset
mnist = datasets.MNIST(
    root='./data/', train=True, transform=img_transform, download=True)
# Data loader
dataloader = torch.utils.data.DataLoader(
    dataset=mnist, batch_size=batch_size, shuffle=True)



for k in set_sigma:
    sigma = copy.deepcopy(k)

    if not os.path.exists('./loss_std{}_secclip{}'.format(sigma,C2)):    
        os.mkdir('./loss_std{}_secclip{}'.format(sigma,C2))
    
    D = discriminator()
    G = generator()
    if torch.cuda.is_available():
        D = D.cuda()
        G = G.cuda()
    # Binary cross entropy loss and optimizer
    criterion = nn.BCELoss()
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)
    
    
    # Start training
    g_loss_list, d_loss_list = [], []
    for epoch in range(num_epoch):
        for i, (img, _) in enumerate(dataloader):
            
            noise=get_noise(sigma)
            
            num_img = img.size(0)
            # =================train discriminator
            img = img.view(num_img, -1)
            real_img = Variable(img).cuda()
            real_label = Variable(torch.ones(num_img)).cuda()
            fake_label = Variable(torch.zeros(num_img)).cuda()
    
            # compute loss of real_img
            real_out = D(real_img)
            d_loss_real = criterion(real_out, real_label)
            
            
            value_d_loss_real = Variable(d_loss_real.float(), requires_grad=False)
            if value_d_loss_real > C1:
                d_loss_real = d_loss_real*value_d_loss_real/C1

            real_scores = real_out  # closer to 1 means better
    
            # compute loss of fake_img
            z = Variable(torch.randn(num_img, z_dimension)).cuda()
            fake_img = G(z)
            fake_out = D(fake_img)
            d_loss_fake = criterion(fake_out, fake_label)
            # d_loss_fake = criterion(fake_out, fake_label)
            fake_scores = fake_out  # closer to 0 means better
    
            # bp and optimize
            d_loss = d_loss_real + d_loss_fake
            
            value_d_loss = Variable(d_loss.float(), requires_grad=False)
            d_loss = d_loss+noise*d_loss/value_d_loss        
            
            d_optimizer.zero_grad()
            d_loss.backward()
            
            d_optimizer.step()
    
            # train generator
            # compute loss of fake_img
            z = Variable(torch.randn(num_img, z_dimension)).cuda()
            fake_img = G(z)
            output = D(fake_img)
            
            g_loss = criterion(output, real_label)
            # value_g_loss = Variable(g_loss.float(), requires_grad=False)
            # g_loss = g_loss*(noise+value_g_loss)/value_g_loss
    
            # bp and optimize
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
    
            if (i + 1) % 200 == 0:
                print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} '
                      'D real: {:.6f}, D fake: {:.6f}'.format(
                          epoch, num_epoch, d_loss.item(), g_loss.item(),
                          real_scores.data.mean(), fake_scores.data.mean()))
        g_loss_list.append(g_loss.item())
        d_loss_list.append(d_loss.item())
        #print('\ng_loss', g_loss_list)
        #print('\nd_loss', d_loss_list)
    with open('./loss_std{}_secclip{}/real_images_digit-{}.txt'.format(sigma,C2,digit),'w',encoding='utf-8') as f:
        f.write('g_loss:\n')
        f.write(str(g_loss_list))
        f.write('\nd_loss:\n')
        f.write(str(d_loss_list))
    plt.figure(1)
    plt.plot(range(len(g_loss_list)), g_loss_list, label='Generator loss')
    plt.plot(range(len(d_loss_list)), d_loss_list, label='Discriminator loss')
    plt.ylabel('train_loss')
    plt.xlabel('num_epoches')
    plt.grid(linestyle = "--")
    #plt.legend()    
    plt.savefig('./loss_std{}_secclip{}/gan_{}.pdf'.format(sigma,C2,sigma))    
#            eval_acc_gan += (fake_label == real_label).float().mean() 
#            print(f'GAN Acc: {eval_acc_gan/(i + 1):.6f}\n') 
#        if epoch == 0:
#            real_images = to_img(real_img.cpu().data)
#            save_image(real_images, './img/real_images.png')
#    
#        fake_images = to_img(fake_img.cpu().data)
#        save_image(fake_images, './img/fake_images-{}.png'.format(epoch + 1))
#    
    torch.save(G.state_dict(), './loss_std{}_secclip{}/generator.pth'.format(sigma,C2))
    torch.save(D.state_dict(), './loss_std{}_secclip{}/discriminator.pth'.format(sigma,C2))
