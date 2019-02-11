import numpy as np

import torch 
import torch as T
import torch.nn as nn

import torchvision

from tqdm import tqdm
from torchvision import datasets, transforms

from utils import ProgrammingNetwork, get_program, train, reg_l1, reg_l2


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


DEVICE = "cuda:0"
PATH = "./models/squeezenet1_0_MNIST"

batch_size = 16
train_loader, test_loader = get_mnist(batch_size)
pretrained_model = torchvision.models.squeezenet1_0(pretrained=True).eval()

input_size = 224
patch_size = 28
ignore_bandwidth = 0

PATH += "_bandwidth_" + str(ignore_bandwidth) + "_"

model = ProgrammingNetwork(
    pretrained_model, input_size, 
    patch_size, blur_sigma=1.5, 
    ignore_bandwidth=ignore_bandwidth, 
    device=DEVICE
)
optimizer = T.optim.Adam([model.p], lr=.05, weight_decay=.96)

nb_epochs = 20
nb_freq = 10
model, loss_history = train(
    model, train_loader, nb_epochs, optimizer,
    C=.05, reg_fun=reg_l2,
    save_freq=nb_freq, 
    save_path=PATH, test_loader=test_loader, device=DEVICE
)


program = get_program(model, PATH, imshow=True)
