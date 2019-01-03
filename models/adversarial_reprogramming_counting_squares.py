import numpy as np

import numpy as np

import torch 
import torch as T
import torch.nn as nn

import torchvision

from tqdm import tqdm
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from random import shuffle
import numpy as np
import scipy
import scipy.misc as misc

#36 
#4


def create_patchv1(nb_square, patch_size=36, square_size=4, border=True):

    mini_patch = torch.zeros((square_size, square_size))

    tmp =  [(np.random.choice(square_size), np.random.choice(square_size))]
    for _ in range(nb_square - 1):
        x = (np.random.choice(square_size), np.random.choice(square_size))
        while x in tmp:
            x = (np.random.choice(square_size), np.random.choice(square_size))
        tmp.append(x)
    
    tmp = list(zip(*tmp))

    mini_patch[np.array(tmp[0]), np.array(tmp[1])] = 1 
    patch = torch.zeros((patch_size, patch_size))

    slice = int(patch_size/square_size)

    for i, j in zip(tmp[0], tmp[1]):
        patch[i*9 : i*9 + slice - border, j*9 : j*9 + slice - border] = 1

    return patch

def create_patch(nb_square, patch_size=36, square_size=4, border=True):

    patch = [1] * nb_square + [0] * (16 - nb_square)
    shuffle(patch)
    patch = np.array(patch).reshape(square_size, square_size)
    patch = scipy.misc.imresize(patch, (patch_size, patch_size), interp='nearest') // 255

    if border :
        for i in range(0, patch_size, patch_size // square_size):
            patch[i, :] = 0.
            patch[:, i] = 0.

    return torch.Tensor(patch).float()


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
        self.mask = get_mask(patch_size, input_size, channel_out, batch_size=1)[0]
        self.input_size = input_size
        self.mask.requires_grad = False

    def forward(self, x):
        #P = tanh (W + M)
        P = nn.Tanh()((1 - self.mask) * self.p) 
        #Xadv = hf (˜x; W) = X˜ + P
        x_adv = x_to_X(x, self.input_size, self.p.shape[0]) + P
        return self.model(x_adv)


 
batch_size = 16
train_loader, test_loader = get_mnist(batch_size)

pretrained_model = torchvision.models.resnet101(pretrained=True).eval()
input_size = 224
patch_size = 28

model = ProgrammingNetwork(pretrained_model, input_size, patch_size)
loss_function = nn.CrossEntropyLoss()
optimizer = T.optim.Adam([model.p])

PATH = "./models/resnet101_mnist.pth"
nb_epochs = 10
loss_history = []

for epoch in range(nb_epochs): 
    for i, (x, y) in tqdm(enumerate(train_loader)):
        y_hat = model(x)
        optimizer.zero_grad()
        loss = loss_function(y_hat, y)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if not i % 10: #save each 10 batches
            T.save(model.state_dict(), PATH)


program = get_program(model, PATH, imshow=True)

