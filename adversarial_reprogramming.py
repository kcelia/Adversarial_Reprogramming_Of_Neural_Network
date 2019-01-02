import torch 
import torch as T
import torch.nn as nn

import torchvision

inception = torchvision.models.resnet101(pretrained=True).eval()#torchvision.models.ResNet101(pretrained=True).eval()#.inception_v3(pretrained=True).eval()

train = torchvision.datasets.MNIST("./data/MNIST", train=True, download=True)
test  = torchvision.datasets.MNIST("./data/MNIST", train=False, download=True)


def x_to_X(x, X_size, channel_out=3):
    """
    The mask for MNIST is ... #TODO

    :param x: batch x, tensor of shape : batch_size*channels*im_size* im_size
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
    ones = T.ones((batch_size, channel_out, patch_size, patch_size))
    return x_to_X(ones, X_size, channel_out)


from torch.functional import F

class ProgrammingNetwork(nn.Module):
    def __init__(self, pretained_model, input_size, patch_size, channel_out=3):
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
        return self.model(x_adv)#[:, :10]

    
model = ProgrammingNetwork(inception, 224, 28)
loss_function = nn.CrossEntropyLoss()
optimizer = T.optim.Adam([model.p]) #model freezé !!
epochs = 1


y_hat = model(T.rand((10, 3, 28, 28)))
optimizer.zero_grad()
loss = loss_function(y_hat, T.tensor([1] * 10).long())
loss.backward()
optimizer.step()





