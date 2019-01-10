import torchvision

from utils import (
    ProgrammingNetwork, standard_normalization, 
    tanh_scaler, program_visualisation
)

path1 = './models\\squeezenet1_0_MNIST_0b_0e.pth'
path2 = './models\\squeezenet1_0_MNIST_0b_40e.pth'

pretrained_model = torchvision.models.squeezenet1_0(pretrained=True).eval()
input_size = 224
patch_size = 28

model = ProgrammingNetwork(pretrained_model, input_size, patch_size)

a, b, c = program_visualisation(model, path1, path2, norm=standard_normalization, imshow=True)
a, b, c = program_visualisation(model, path1, path2, norm=tanh_scaler, imshow=True)
 
