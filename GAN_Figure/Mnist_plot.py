# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 04:46:28 2020

@author: WEIKANG
"""

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import math
import torch
import copy
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable


def location_excha(images, x, y):
    
    imgs = copy.deepcopy(images[y])
    images[y] = copy.deepcopy(images[x])  
    images[x] = copy.deepcopy(imgs)  
    
    return images

def plot_images10(images, num_row, set_epsilon, smooth=True):
    # Interpolation type.
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    # Create figure with sub-plots.
    fig, axes = plt.subplots(num_row, 10)

    # Adjust vertical spacing.
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    # For each entry in the grid.
    for i, ax in enumerate(axes.flat):
        # Get the i'th image and only use the desired pixels.
        img = images[i, :, :]
        
        # Plot the image.
        ax.imshow(img, interpolation=interpolation, cmap='binary')

        # Show true and predicted classes.
        if i == 0:
            ylabel = "Real Data"
            ax.set_ylabel(ylabel)            
        elif i == 10:
            ylabel = "(a)"
            ax.set_ylabel(ylabel)            
        elif i == 20:
            ylabel = "(b)"
            ax.set_ylabel(ylabel)            
        elif i == 30:
            ylabel = "(c)"
            ax.set_ylabel(ylabel)            
        elif i == 40:
            ylabel = "(d)"
            ax.set_ylabel(ylabel)
        # Show the classes as the label on the x-axis.
            ax.set_ylabel(ylabel)
            
        # Remove ticks.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
    
def unique_index(L,f):
    return [i for (i,value) in enumerate(L) if value==f]
    
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):

        image, label = self.dataset[int(self.idxs[item])]
        return image, label   
    
def data_rank(dataset):
    num_labels = dataset.train_labels.numpy()
    classes = np.unique(num_labels)
    classes_index = []
    for m in range(len(classes)):
        classes_index.append(unique_index(num_labels, classes[m]))
    return classes_index, classes
    
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
    
if __name__ == '__main__':  
    
    batch_size = 80
    z_dimension = 100
    set_sigma_noise = [-1, 1.0, 10]
    set_sigma_para = [5.0, 50]
    set_epsilon = [-1, 1, 10, 1, 10]
    num_row = len(set_sigma_noise)+len(set_sigma_para)
    G = generator()
    if torch.cuda.is_available():
        G = G.cuda()
    
    # Image processing
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # MNIST dataset
    Mnist = datasets.MNIST(
        root='../data/', train=True, transform=img_transform, download=True)
    # Data loader
    dataloader = torch.utils.data.DataLoader(
        dataset=Mnist, batch_size=batch_size, shuffle=True)
    
    FashionMnist = datasets.FashionMNIST(
    root='../data', train=True, transform=transforms.ToTensor(), download=True)

    
    # Mnist classify
    classes_index, classes = data_rank(Mnist)
    
    #images = mnist[classes_index[0][0]][0]
    images = []
    z = Variable(torch.randn(batch_size, z_dimension)).cuda()
    for sigma in set_sigma_noise:
        for m in range(len(classes)):
            if sigma == -1:
                # Minst dataset
                img = np.array(Mnist[classes_index[m][0]][0])
                img = img.squeeze()
                images.append(img)
            else:
                # Generated data with gaussian noise (sigma = 10)
                G.load_state_dict(torch.load('./loss_std{}/'.format(sigma) + 'generator_digit-{}.pth'.format(m)))
                fake_img, _, _ = G(z)
                fake_images = to_img(fake_img.cpu().data)
                img = fake_images[0][0]
                img = np.array(img)
                
                img = img.squeeze()
                images.append(img)
    for sigma in set_sigma_para:
        for m in range(len(classes)):
                G.load_state_dict(torch.load('./para_std{}/'.format(sigma) + 'generator_digit-{}.pth'.format(m)))
                fake_img, _, _ = G(z)
                fake_images = to_img(fake_img.cpu().data)
                img = fake_images[0][0]
                img = np.array(img)
                
                img = img.squeeze()
                images.append(img)            
        
    images = np.array(images)
    
    plot_images10(images=images, num_row = num_row, set_epsilon= set_epsilon)