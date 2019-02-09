import numpy as np

import torch 
import torch as T

import torchvision

from torchvision import datasets, transforms

from tqdm import tqdm

from utils import get_program, tanh_scaler, ProgrammingNetwork, run_test_accuracy


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

def make_programming_network(pretrained_model, input_size=224, patch_size=28, device="cpu"):
    model = ProgrammingNetwork(pretrained_model, input_size, patch_size, blur_sigma=1.5, device=device)
    return model

def run_test_accuracy(model, test_loader, device='cpu'):
    test_accuracy = []
    for i, (x, y) in enumerate(tqdm(test_loader)):
        y_hat = model(x)
        (y_hat.argmax(1).to(device) == y).float()
        test_accuracy.extend((y_hat.argmax(1).to(device) == y).float().numpy())

    return np.array(test_accuracy).mean()

def prune_program(model, path, band_width, band_value=0, batch_size=16, location="cpu", device='cpu'):
    #profondeur,  largeur, hauteur 
    p = torch.load(path, map_location=location)
    p[:, 0: band_width, :] = p[:, -band_width:, :] = band_value
    p[:, :, 0: band_width] = p[:, :, -band_width:] = band_value

    new_p = torch.autograd.Variable(torch.tensor(p), requires_grad=True) #torch.tensor(..).float()
    model.p = new_p
    model.p.requires_grad = False #eval mode

    _, test_loader = get_mnist(batch_size)
    return run_test_accuracy(model, test_loader, device)

def to_device(programmingNetwork, device="cpu"):
    programmingNetwork.device = device
    programmingNetwork.p = T.autograd.Variable(
        programmingNetwork.p.to(device),
        requires_grad=True
    )
    programmingNetwork.model.to(device)
    return programmingNetwork

DEVICE = "cpu"
PATH = "a.pth"
pretrained_model = torchvision.models.squeezenet1_0(pretrained=True).eval()
model = make_programming_network(pretrained_model, device=DEVICE)


bands_width = list(range(1, 11)) + list(range(15, 51, 5)) + [100, 112]
band_value = 0


test_pruning_accuracy = {band_width: to_device(prune_program(model, PATH, band_width, band_value=0)) for band_width in bands_width }

np.save("./models/MNIST_Squeeze1_0_test_pruning_accuracy", test_pruning_accuracy)



