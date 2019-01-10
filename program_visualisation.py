import torchvision

from utils import (
    ProgrammingNetwork, standard_normalization, 
    tanh_scaler, program_visualisation
)

path1 = './models\\squeezenet1_0_MNIST_0b_0e.pth'
path2 = './models\\squeezenet1_0_MNIST_0b_130e.pth'


a, b, c = program_visualisation(path1, path2, norm=standard_normalization, imshow=True)
a, b, c = program_visualisation(path1, path2, norm=tanh_scaler, imshow=True)
 
