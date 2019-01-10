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
from scipy.ndimage import gaussian_filter


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

def get_program(path, imshow=False):
    """
    This function return the program P as a numpy and displays it 
    according to the value of the imshow's attribut 

    :param path: the path that contains the saved weights
    :param imshow: a boolean, if it's true then we display the weights P

    :type programming_network: ProgrammingNetwork
    :type path: str
    :type imshow: bool
    
    :return: img 
    :rtype: numpy.ndarray
    """
    program = torch.load(path)

    img = program.detach().permute(1, 2, 0).numpy()
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

    def __init__(self, pretained_model, input_size, patch_size, channel_out=3, blur_sigma=0., device="cpu"):
        """
        Constructor

        :param pretrained_model: the model to hitjak
        :param input_size: the img's size excepected by pretrained_model
        :param patch_size: the size of the small target domain img
        :param channel_out: nb channel
        :param blur_sigma: 0 if no bluring else the sigma used to blur the program before training
        :param device: device used for training
        
        :type pretrained_model: modul
        :type input_size: int
        :type patch_size: int
        :type channel_out: int
        :type blur_sigma: float
        :type device: str
        """
        super().__init__()
        self.device = device
        self.blur_sigma = blur_sigma
        self.model = pretained_model.to(self.device)
        self.p = T.randn((channel_out, input_size, input_size))
        if blur_sigma:
            program = self.p.to("cpu").detach().permute(1, 2, 0).numpy()
            program = gaussian_filter(program, self.blur_sigma)
            program = T.tensor(program).float().permute(2, 0, 1)
            self.p = program
        self.mask = get_mask(patch_size, input_size, channel_out, batch_size=1)[0]
        self.p = T.autograd.Variable((self.p * (1 - self.mask)).to(self.device), requires_grad=True)
        self.mask = self.mask.to(self.device)
        self.one = T.tensor(1.).to(self.device)
        self.input_size = input_size
        self.mask.requires_grad = False

    def forward(self, x):
        #P = tanh (W + M)
        P = nn.Tanh()((self.one - self.mask) * self.p) 
        #Xadv = hf (˜x; W) = X˜ + P
        x_adv = x_to_X(x, self.input_size, self.p.shape[0]).to(self.device) + P
        return self.model(x_adv)

def reg_l1(x):
    """
    This function is a L1 regularisation for a given matrix x

    :param x: the matrix to regularize

    :type x: torch.Tensor

    :return: the l1 norm of x
    :rtype: torch.Tensor
    """
    return x.abs().mean()

def reg_l2(x):
    """
    This function is a L2 regularisation for a given matrix x

    :param x: the matrix to regularize

    :type x: torch.Tensor

    :return: the l2 norm of x
    :rtype: torch.Tensor
    """
    return (x ** 2).mean()

def train(model, train_loader, nb_epochs, optimizer, C=0., reg_fun=None, save_freq=100, save_path="./models/", test_loader=None, device="cpu"):
    """
    This function is used to train our adversarial program

    :param model: the model to train
    :param train_loader: train loader, in our case it can be MNIST_dataset, Shuffled_MNIST_dataset, Counting_squares_dataset
    :param nb_epochs: numbre of epochs 
    :param optimizer: the otpimizer
    :param C: the factor regularization
    :param reg_fun: the regularization (should be pytorch graph friendly) put to None if no regularization
    :param save_freq: the state of our model will be saved each "save_freq" times 
    :param save_path: the state of our model that will be saved each  "save_freq" times in a path called save_path
    :param test_loader: specify if we want to get the test accuracy after each epoch else None
    :param device: the device used for training 

    :type model: ProgrammingNetwork
    :type train_loader: torch.utils.data.dataloader.DataLoader
    :type nb_epochs: int
    :type optimizer: torch.optim
    :type C: float
    :type reg_fun: function or None
    :type save_freq: int
    :type save_path: str
    :type test_loader: torch.utils.data.dataloader.DataLoader
    :type device: str

    :return: the model modified and a list of the training loss
    :rtype: tuple(ProgrammingNetwork, list)
    """
    loss_history = []
    test_accuracy_history = []
    loss_function = nn.CrossEntropyLoss()
    for epoch in range(nb_epochs): 
        for i, (x, y) in enumerate(tqdm(train_loader)):
            y_hat = model(x)
            optimizer.zero_grad()
            loss = loss_function(y_hat, y.to(device)) + (C * reg_fun(model.p) if reg_fun else 0.)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            if not i % save_freq:
                T.save(model.p, save_path + "_{}b_{}e.pth".format(epoch, i))
                np.save(save_path + "_loss_history", loss_history)            
        if test_loader:
            test_accuracy_history.append(run_test_accuracy(model, test_loader))
            np.save(save_path + "_test_accuracy_history", test_accuracy_history)
    return model, loss_history

def run_test_accuracy(model, test_loader):
    """
    This function compute the accuracy for a given model

    :param model: the model to evaluate
    :param test_loader: the dataloader to evaluate on

    :type model: ProgrammingNetwork
    :type test_loader: torch.utils.data.dataloader.DataLoader

    :return: the accuracy of the model
    :rtype: float
    """
    test_accuracy = []
    for i, (x, y) in enumerate(tqdm(test_loader)):
        y_hat = model(x)
        (y_hat.argmax(1).to('cpu') == y).float()
        test_accuracy.extend((y_hat.argmax(1).to('cpu') == y).float().numpy())

    return np.array(test_accuracy).mean()

def standard_normalization(matrix):
    """
    This function normalize the value of the given matrix between 0 and 1.
    This allows us to visualize the weigths of the pregram before the tanh
    clipping and getting into the network.

    :param matrix: the matrix to normalize

    :type matrix: torch.Tensor

    :return: the matrix normalized
    :rtype: torch.Tensor
    """
    minimum = matrix.min()
    abs_min = np.sign(minimum) * minimum
    return (matrix + abs_min) / (matrix.max() + abs_min)

def tanh_scaler(matrix):
    """
    This function scaled the value of the given matrix between 0 and 1 
    by using tanh then a rescale. This is the closest form of visualization
    to the real program given as input.
    
    :param matrix: the matrix to normalize

    :type matrix: torch.Tensor

    :return: the matrix normalized
    :rtype: torch.Tensor
    """    
    return (np.tanh(matrix) + 1) / 2.

def program_visualisation(path1, path2, norm=standard_normalization, imshow=False):
    """
    This function is used for visualizing 2 programs and the difference between 2 programs

    :param path1: the path to a first version of the pytorch saved program
    :param path2: the path to a second version of the pytorch saved program
    :param norm: the function to use for rescaling program values between [0, 1]
    :param imshow: display or not the programs

    :type path1: str
    :type path2: str
    :type norm: function
    :type imshow: bool

    :return: the tuple of the 2 programs and difference ready for visualization
    :rtype: tuple[torch.Tensor]
    """
    img1 = get_program(path1, imshow=False)
    img2 = get_program(path2, imshow=False)
    diff = img2 - img1
    images = list(map(norm, (img1, img2))) + [standard_normalization(diff)]

    if imshow:
        fig = plt.figure(figsize=(10, 10))
        columns, rows, j = len(images), 1, 0
        titles = ["Program1", "Program2", "Difference"]
        for i in range(1, columns * rows + 1):
            fig.add_subplot(rows, columns, i)
            plt.title(titles[j])
            plt.imshow(images[j])
            j += 1
        fig.suptitle(str(norm).split()[1].replace('_', ' '), fontsize=15)
        plt.show()

    return img1, img2, diff

