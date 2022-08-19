import pickle
from random import shuffle 
from PIL import Image 
from torch import nn 
import torch 
from torchvision import datasets, transforms 
from torch.utils.data import Dataset 
import skimage.io as io
from PIL import Image


import os 


def image_preprocess_transform():
    pretrained_size = 224
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
                               transforms.Resize(pretrained_size),
                               transforms.RandomRotation(5),
                               transforms.RandomHorizontalFlip(0.5),
                               transforms.RandomCrop(pretrained_size, padding=10),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=pretrained_means,
                                                    std=pretrained_stds)
                        ])

    test_transform = transforms.Compose([
                               transforms.Resize(pretrained_size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=pretrained_means,
                                                    std=pretrained_stds)
                        ]) 
    return train_transform, test_transform 



def cifa10_data_load(data_path='data/cifar', batch_size=8, distribution=False):
    # image transform 
    train_transform, test_transform = image_preprocess_transform() 

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck') 

    train_set = datasets.CIFAR10(data_path, train=True, download=False, transform=train_transform)
    
    test_set = datasets.CIFAR10(data_path, train=False, download=False, transform=test_transform)

    if distribution == False:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                  shuffle=True) 
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                                  shuffle=False) # num_workers=2 
    else: 
        train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set)
        val_sampler = torch.utils.data.sampler.SequentialSampler(test_set) 
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=8,
                                                  sampler=train_sampler) 
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=8,
                                                  sampler=val_sampler) 

    return train_loader, test_loader


def cifa100_data_load(data_path='data/cifar', batch_size=8, distribution=False):
    # image transform 
    train_transform, test_transform = image_preprocess_transform() 


    train_set = datasets.CIFAR100(data_path, train=True, download=False, transform=train_transform)
    
    test_set = datasets.CIFAR100(data_path, train=False, download=False, transform=test_transform)

    if distribution == False:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                  shuffle=True) 
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                                  shuffle=False) # num_workers=2 
    else: 
        train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set)
        val_sampler = torch.utils.data.sampler.SequentialSampler(test_set) 
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=8,
                                                  sampler=train_sampler) 
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=8,
                                                  sampler=val_sampler) 

    return train_loader, test_loader




def imagenet_data_load(data_path='data/imagenet', batch_size=8): 
    train_transform, test_transform = image_preprocess_transform() 

    train_set = datasets.ImageNet(data_path, split='train', transform=train_transform, download=False) 
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True) 

    val_set = datasets.ImageNet(data_path, split='val', transform=test_transform, download=False) 
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False) 

    return train_loader, val_loader 




def data_plot():
    cifar10_path = 'data/cifar-10/data_batch_1' 
    with open(cifar10_path, 'rb') as f: 
        dict = pickle.load(f, encoding='bytes') 

    img = dict[b'data'][1].reshape(3, 32, 32).transpose(1,2,0)
    im = Image.fromarray(img) 
    im.show() 



