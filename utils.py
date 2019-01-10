import numpy as np

import torch 
import torch as T
import torch.nn as nn

import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset

from tqdm import tqdm
from random import shuffle

import matplotlib.pyplot as plt

import scipy
import scipy.misc as misc


def x_to_X(x, X_size, channel_out=3):
    """
    This function places a batch of small image x in the center 
    of a bigger one of size X_size with zero padding.

    :param x: batch x, [batch_size, channels, im_size, im_size]
    :param X_size: the size of the new image 
    :param channel_out: the number of the channel

    :type x: torch.Tensor
    :type X_size: int 
    :type channel_out: int

    :return: x centred in X_size zerroed image 
    :rtype: torch.tensor
    """
    X = T.zeros((x.shape[0], channel_out, X_size, X_size))

    start_x = X_size // 2 - x.shape[2] // 2
    end_x = start_x + x.shape[2] 
    start_y = X_size // 2 - x.shape[3] // 2
    end_y = start_y + x.shape[3]

    x = x.expand(x.shape[0], channel_out, x.shape[2], x.shape[3])
    X[:, :, start_x:end_x, start_y:end_y] = x

    return X

def get_mask(patch_size, X_size, channel_out, batch_size=1):
    """
    This function return the mask for an img of size patch_size 
    which is in the center of a bigger on with size X_size

    :param patch_size: the size of patch that we want to put in the center
    :param X_size: the new size of the img
    :param channel_out: nb channels
    :param batch_size: nb times that the mask will be replicated

    :type patch_size: int
    :type X_size: int
    :type channel_out: int
    :type batch_size: int
    
    :return: binary mask
    :rtype: torch.Tensor 
    """
    ones = T.ones((batch_size, channel_out, patch_size, patch_size))
    return x_to_X(ones, X_size, channel_out)

def get_program(programming_network, path, imshow=False):
    """
    This function return the program P as a numpy and displays it 
    according to the value of the imshow's attribut 

    :param programming_network: model that we want to have its weights
    :param path: the path that contains the saved weights
    :param imshow: a boolean, if it's true then we display the weights P

    :type programming_network: ProgrammingNetwork
    :type path: str
    :type imshow: bool
    
    :return: img 
    :rtype: numpy.ndarray
    """

    programming_network.load_state_dict(torch.load(path))
    programming_network.eval()

    img = programming_network.p.detach().permute(1, 2, 0).numpy()
    if imshow:
        plt.imshow(img)
        plt.show()
    return img

class ProgrammingNetwork(nn.Module):
    """
    This class is the module that contains the network
    that will be uilized and the associated programm 
    that will be learned to hijak the first one
    """

    def __init__(self, pretained_model, input_size, patch_size, channel_out=3, device="cpu"):
        """
        Constructor

        :param pretrained_model: the model to hitjak
        :param input_size: the img's size excepected by pretrained_model
        :param patch_size: the size of the small target domain img
        :param channel_out: nb channel
        :param device: device used for training
        
        :type pretrained_model: modul
        :type input_size: int
        :type patch_size: int
        :type channel_out: int
        :type device: str
        """
        super().__init__()
        self.device = device
        self.model = pretained_model.to(self.device)
        self.p = T.autograd.Variable(T.randn((channel_out, input_size, input_size)).to(self.device), requires_grad=True)
        self.mask = get_mask(patch_size, input_size, channel_out, batch_size=1)[0]
        self.input_size = input_size
        self.mask.requires_grad = False

    def forward(self, x):
        #P = tanh (W + M)
        P = nn.Tanh()((1 - self.mask) * self.p) 
        #Xadv = hf (˜x; W) = X˜ + P
        x_adv = x_to_X(x, self.input_size, self.p.shape[0]).to(self.device) + P
        return self.model(x_adv)

def train(model, train_loader, nb_epochs, optimizer, save_freq=100, save_path="./models/", device="cpu"):
    """
    This function is used to train our adversarial program

    :param model: the model to train
    :param train_loader: train loader, in our case it can be MNIST_dataset, Shuffled_MNIST_dataset, Counting_squares_dataset
    :param nb_epochs: numbre of epochs 
    :param optimizer: the otpimizer
    :param save_freq: the state of our model will be saved each "save_freq" times 
    :param save_path: the state of our model that will be saved each  "save_freq" times in a path called save_path
    :param device: the device used for training 

    :type model: ProgrammingNetwork
    :type train_loader: torch.utils.data.dataloader.DataLoader
    :type nb_epochs: int
    :type optimizer: torch.optim
    :type save_freq: int
    :type save_path: str
    :type device: str

    :return: the model modified and a list of the training loss
    :rtype: tuple(ProgrammingNetwork, list)
    """
    loss_history = []
    loss_function = nn.CrossEntropyLoss()
    for epoch in range(nb_epochs): 
        for i, (x, y) in enumerate(tqdm(train_loader)):
            y_hat = model(x)
            optimizer.zero_grad()
            loss = loss_function(y_hat, y.to(device))
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            if not i % save_freq: #save each save_freq batches
                T.save(model.state_dict(), save_path + "_{}b_{}e.pth".format(epoch, i))
    return model, loss_history
