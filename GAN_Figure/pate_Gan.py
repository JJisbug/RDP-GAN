########## This code is for the pate-gan, which can generate DP guranteed MNIST data. We set 5 student-discriminator and 1 teacher-discriminator
import numpy as np
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
if not os.path.exists('./change_uniform_1000'):
    os.mkdir('./change_uniform_1000')


def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 28, 28)
    return out
def get_noise():
    Noise=np.random.normal(0,0.01)
    return Noise 

batch_size = 128
num_epoch = 500
z_dimension = 100

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

class classfier(nn.Module):
    def __init__(self):
        super(classfier, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2), nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.dis(x)
        return x
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

C = classfier()
D1 = discriminator()
D2 = discriminator()
D3 = discriminator()
D4 = discriminator()
D5 = discriminator()
G = generator()
if torch.cuda.is_available():
    C = C.cuda()
    D1 = D1.cuda()
    D2 = D2.cuda()
    D3 = D3.cuda()
    D4 = D4.cuda()
    D5 = D5.cuda()
    G = G.cuda()
# Binary cross entropy loss and optimizer
criterion = nn.BCELoss()
c_optimizer = torch.optim.Adam(C.parameters(), lr=0.0003)
d1_optimizer = torch.optim.Adam(D1.parameters(), lr=0.0003)
d2_optimizer = torch.optim.Adam(D2.parameters(), lr=0.0003)
d3_optimizer = torch.optim.Adam(D3.parameters(), lr=0.0003)
d4_optimizer = torch.optim.Adam(D4.parameters(), lr=0.0003)
d5_optimizer = torch.optim.Adam(D5.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)

# Start training
for epoch in range(num_epoch):
    for i, (img, _) in enumerate(dataloader):
        num_img = img.size(0)
        
        # =================train discriminator_1
        img = img.view(num_img, -1)
        real_img = Variable(img).cuda()
        real_label = Variable(torch.ones(num_img)).cuda()
        fake_label = Variable(torch.zeros(num_img)).cuda()
        fake_scores_1 = Variable(torch.zeros(num_img)).cuda()
        # compute loss of real_img
        real_out_1 = D1(real_img)
        d_loss_real_1 = criterion(real_out_1, real_label)
        real_scores_1 = real_out_1  # closer to 1 means better
        # compute loss of fake_img
        z = Variable(torch.randn(num_img, z_dimension)).cuda()
        z1 = z
        z2 = z
        fake_img = G(z)
        fake_out_1 = D1(fake_img)
        d_loss_fake_1 = criterion(fake_out_1, fake_label)
        for j in range(num_img):
            if fake_out_1[j] > 0.5:
                fake_scores_1[j] = 1  # closer to 0 means better
            else:
                fake_scores_1[j] = 0      
        # bp and optimize
        d_loss_1 = d_loss_real_1 + d_loss_fake_1
        d1_optimizer.zero_grad()
        d_loss_1.backward()
        d1_optimizer.step()
        
        # =================train discriminator_2
        fake_scores_2 = Variable(torch.zeros(num_img)).cuda()
        # compute loss of real_img
        real_out_2 = D2(real_img)
        d_loss_real_2 = criterion(real_out_2, real_label)
        real_scores_2 = real_out_2  # closer to 1 means better
        # compute loss of fake_img
        fake_img = G(z)
        fake_out_2 = D2(fake_img)
        d_loss_fake_2 = criterion(fake_out_2, fake_label)
        for j in range(num_img):
            if fake_out_2[j] > 0.5:
                fake_scores_2[j] = 1  # closer to 0 means better
            else:
                fake_scores_2[j] = 0      
        # bp and optimize
        d_loss_2 = d_loss_real_2 + d_loss_fake_2
        d2_optimizer.zero_grad()
        d_loss_2.backward()
        d2_optimizer.step()
        
        # =================train discriminator_3
        fake_scores_3 = Variable(torch.zeros(num_img)).cuda()
        # compute loss of real_img
        real_out_3 = D3(real_img)
        d_loss_real_3 = criterion(real_out_3, real_label)
        real_scores_3 = real_out_3  # closer to 1 means better
        # compute loss of fake_img
        fake_img = G(z)
        fake_out_3 = D3(fake_img)
        d_loss_fake_3 = criterion(fake_out_3, fake_label)
        for j in range(num_img):
            if fake_out_3[j] > 0.5:
                fake_scores_3[j] = 1  # closer to 0 means better
            else:
                fake_scores_3[j] = 0      
        # bp and optimize
        d_loss_3 = d_loss_real_3 + d_loss_fake_3
        d3_optimizer.zero_grad()
        d_loss_3.backward()
        d3_optimizer.step()
        
        # =================train discriminator_4
        fake_scores_4 = Variable(torch.zeros(num_img)).cuda()
        # compute loss of real_img
        real_out_4 = D4(real_img)
        d_loss_real_4 = criterion(real_out_4, real_label)
        real_scores_4 = real_out_4  # closer to 1 means better
        # compute loss of fake_img
        fake_img = G(z)
        fake_out_4 = D4(fake_img)
        d_loss_fake_4 = criterion(fake_out_4, fake_label)
        for j in range(num_img):
            if fake_out_4[j] > 0.5:
                fake_scores_4[j] = 1  # closer to 0 means better
            else:
                fake_scores_4[j] = 0      
        # bp and optimize
        d_loss_4 = d_loss_real_4 + d_loss_fake_4
        d4_optimizer.zero_grad()
        d_loss_4.backward()
        d4_optimizer.step()  
        
        # =================train discriminator_5
        fake_scores_5 = Variable(torch.zeros(num_img)).cuda()
        # compute loss of real_img
        real_out_5 = D5(real_img)
        d_loss_real_5 = criterion(real_out_5, real_label)
        real_scores_5 = real_out_5  # closer to 1 means better
        # compute loss of fake_img
        fake_img = G(z)
        fake_out_5 = D5(fake_img)
        d_loss_fake_5 = criterion(fake_out_5, fake_label)
        for j in range(num_img):
            if fake_out_5[j] > 0.5:
                fake_scores_5[j] = 1  # closer to 0 means better
            else:
                fake_scores_5[j] = 0      
        # bp and optimize
        d_loss_5 = d_loss_real_5 + d_loss_fake_5
        d5_optimizer.zero_grad()
        d_loss_5.backward()
        d5_optimizer.step()
        
        # majority vote and perturbation

        fake_scores=fake_scores_1+fake_scores_2+fake_scores_3+fake_scores_4+fake_scores_5
        n= Variable(torch.randn(num_img)).cuda()
        fake_scores=fake_scores+n
        for j in range(num_img):       
            if fake_scores[j] > 2.5:
                fake_scores[j] = 1
            else:
                fake_scores[j] = 0                
        
        #===================train classifier
        #z = Variable(torch.randn(num_img, z_dimension)).cuda()
#        a=z
        fake_img = G(z1)
        fake_out_c =  C(fake_img)
        real_out_c = C(real_img)
        d_loss_real_c = criterion(real_out_c, real_label)
        c_loss = criterion(fake_out_c, fake_scores)+d_loss_real_c
        c_optimizer.zero_grad()
        c_loss.backward()
        c_optimizer.step() 
       
        # ===============train generator
        # compute loss of fake_img
#        if epoch < 1:
#        z = Variable(torch.randn(num_img, z_dimension)).cuda()
#        else: 
#           z = Variable(torch.normal(0,2, [num_img, z_dimension])).cuda()
#        fake_img = G(z)
#        output = D(fake_img)
        #z = Variable(torch.randn(num_img, z_dimension)).cuda()
        fake_img = G(z2)     
        fake_out_c_new =  C(fake_img)        
        g_loss = criterion(fake_out_c_new, real_label)
        # bp and optimize
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        
       

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], c_loss: {:.4f}, g_loss: {:.4f}, d_loss_1: {:.4f}, d_loss_5: {:.4f}'.format(
                    epoch, num_epoch, c_loss.item(), g_loss.item(), d_loss_1.item(), d_loss_5.item() ))
            
    if epoch == 0:
          real_images = to_img(real_img.cpu().data)
          save_image(real_images, './pate/real_images.png')
          
#fake_images = to_img(fake_img.cpu().data)
#save_image(fake_images, './pate/fake_images-{}.png'.format(epoch + 1))
#num_ite=10
#for ite in range(num_ite):    
##    z = Variable(torch.normal(0,1, [num_img, z_dimension])).cuda()
##    noise=Variable(torch.normal(0,sigma, [num_img, z_dimension])).cuda()
#    z = Variable(torch.rand(num_img, z_dimension)).cuda()
#    #z=z+noise
#num_ite=10
#for ite in range(num_ite):      
#    fake_img = G(z)
#    fake_images = to_img(fake_img.cpu().data)
#    save_image(fake_images, './pate/fake_images-{}.png'.format(ite + 1))

    if epoch == 0:
        real_images = to_img(real_img.cpu().data)
        save_image(real_images, './pate/real_images.png')

    fake_images = to_img(fake_img.cpu().data)
    save_image(fake_images, './pate/fake_images-{}.png'.format(epoch + 1))

torch.save(G.state_dict(), './generator.pth')
torch.save(C.state_dict(), './discriminator.pth')
torch.save(D1.state_dict(), './discriminator1.pth')
torch.save(D2.state_dict(), './discriminator2.pth')
torch.save(D3.state_dict(), './discriminator3.pth')
torch.save(D4.state_dict(), './discriminator4.pth')
torch.save(D5.state_dict(), './discriminator5.pth')
