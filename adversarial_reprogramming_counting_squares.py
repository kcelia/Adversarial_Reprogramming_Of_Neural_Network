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

from utils import ProgrammingNetwork, get_program, train, reg_l1, reg_l2


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
    """
    :param nb_square: the number of squares to set on
    :param patch_size: the size of the patch wich will be in the center of the img
    :param square_size: the size of the squares <!> square_size < patch_size
    :param border: a boolean, that allows to separate the squares (to have a better display)
    
    :type nb_square: int
    :type patch_size: int
    :type square_size: int
    :type border: bool

    :return: the new img 
    :rtype: torch.tensor
    """
    patch = [1] * nb_square + [0] * (16 - nb_square)
    shuffle(patch)
    patch = np.array(patch).reshape(square_size, square_size)
    patch = scipy.misc.imresize(patch, (patch_size, patch_size), interp='nearest') // 255

    if border:
        for i in range(0, patch_size, patch_size // square_size):
            patch[i, :] = 0.
            patch[:, i] = 0.

    return torch.Tensor([patch]).float()

class SquaresDataset(Dataset):
    def __init__(self, patch_size=36, square_size=4, dataset_size=100000):
        self.patch_size = patch_size
        self.square_size = square_size
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, item):
        y = np.random.randint(1, 10) 
        return (
            create_patch(y, self.patch_size, self.square_size, border=False), 
            T.tensor(y).long()
        )

def get_counting_squares(batch_size, dataset_size=100000): 

    train_loader = T.utils.data.DataLoader(
        SquaresDataset(dataset_size=dataset_size),
        batch_size=batch_size,
        shuffle=True
    )
    
    return train_loader


DEVICE = 'cuda:0'
PATH = "./models/squeezenet1_0_counting_squares_cici_"

batch_size = 16
train_loader = get_counting_squares(batch_size)
test_loader  = get_counting_squares(batch_size, 1000)

pretrained_model = torchvision.models.squeezenet1_0(pretrained=True).eval()

input_size = 224
patch_size = 36

model = ProgrammingNetwork(pretrained_model, input_size, patch_size, blur_sigma=1.5, device=DEVICE)
optimizer = T.optim.Adam([model.p])

nb_epochs = 20
nb_freq = 10
model, loss_history = train(
    model, train_loader, nb_epochs, optimizer,
    C=.05, reg_fun=reg_l2,
    save_freq=nb_freq, 
    save_path=PATH, test_loader=test_loader, device=DEVICE
)

program = get_program(model, PATH, imshow=True)



