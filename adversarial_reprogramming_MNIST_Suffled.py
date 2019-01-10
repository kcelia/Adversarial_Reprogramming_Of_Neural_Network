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



def shuffle_mnist(p, seed=23):
    """
    This function shuffle an image data

    :param p: the image that will be shuffled
    :param seed: we use a random seed to have the same shuffling at each call of the function

    :type p: torch.tensor
    :type seed: int

    :return: a shuffled image
    :rtype: torch.tensor
    """
    lst = [(i,j) for i in  range(p.shape[1]) for j in range(p.shape[2])]
    random.Random(seed).shuffle(lst)

    plan = {
        (i, j): lst[i * p.shape[1] + j] 
        for i in  range(p.shape[1]) for j in range(p.shape[2])
    }

    out = T.zeros(p.shape)
    for ((i, j), (ii, jj)) in plan.items():
        out[:, i, j] = p[:, ii, jj]

    return out

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

def get_mnist(batch_size):
    """
    This function retruns the train and test loader of mnist 
    dataset for a given batch_size

    :param batch_size: size of the batch for data loader
    
    :type batch_size: int

    :return: train and test loader
    :rtype: tuple[torch.utils.data.DataLoader]
    """
    train_loader = T.utils.data.DataLoader(datasets.MNIST(
        './data', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True
    ) 
    test_loader = T.utils.data.DataLoader(datasets.MNIST(
        './data', train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True
    ) 
    return train_loader, test_loader


class ProgrammingShuffledNetwork(nn.Module): #TODO: RENAME FOR SUFFLED VERSION
    """
    This class is the module that contains the network
    that will be uilized and the associated programm 
    that will be learned to hijak the first one
    with a shuffled input
    """

    def __init__(self, pretained_model, input_size, patch_size, channel_out=3):
        """
        Constructor

        :param pretrained_model: the model to hitjak
        :param input_size: the img's size excepected by pretrained_model
        :param patch_size: the size of the small target domain img
        :param channel_out: nb channel
        
        :type pretrained_model: modul
        :type input_size: int
        :type patch_size: int
        :type channel_out: int
        """
        super().__init__()
        self.model = pretained_model
        self.p = T.autograd.Variable(T.randn((channel_out, input_size, input_size)), requires_grad=True)
        self.mask = shuffle_mnist(get_mask(patch_size, input_size, channel_out, batch_size=1)[0])
        self.input_size = input_size
        self.mask.requires_grad = False

    def forward(self, x):
        x = T.tensor([shuffle_mnist(xx).numpy() for xx in x])
        #P = tanh (W + M)
        P = nn.Tanh()((1 - self.mask) * self.p) 
        #Xadv = hf (˜x; W) = X˜ + P
        x_adv = x_to_X(x, self.input_size, self.p.shape[0]) + P
        return self.model(x_adv)

 
batch_size = 16
train_loader, test_loader = get_mnist(batch_size)

pretrained_model = torchvision.models.squeezenet1_0(pretrained=True).eval()

input_size = 224
patch_size = 4

model = ProgrammingShuffledNetwork(pretrained_model, input_size, patch_size)
loss_function = nn.CrossEntropyLoss()
optimizer = T.optim.Adam([model.p])

PATH = "./models/squeezenet1_0_MNIST_shuffled.pth"

nb_epochs = 1
loss_history = []

for epoch in range(nb_epochs): 
    for i, (x, y) in enumerate(tqdm(train_loader)):
        y_hat = model(x)
        optimizer.zero_grad()
        loss = loss_function(y_hat, y)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if not i % 10: #save each 10 batches
            T.save(model.state_dict(), PATH)


program = get_program(model, PATH, imshow=True)
