import torch
import torch.nn as nn
import torch.nn.functional
import numpy as np
from torch.autograd import Variable

def langevin_sampling(beta, Encoder, steps=100, step_size=0.001, sigma=1, initial=None, burn_in=90, batch = 1):
    # initialize
    # noise = lambda x: x + sigma * torch.randn_like(x)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if initial is not None:
        Y = initial.to(device)
    else:
        Y = torch.rand([batch, 3, 32, 32]).to(device)

    # Langevin updates
    Y_samples = None
    for j in range(steps):
        U = sigma * torch.randn_like(Y)
        # U = 0                               # no noise
        Y_in = Variable(Y, requires_grad=True)
        derivative = torch.autograd.grad(torch.dot(Encoder(Y_in).view(-1), beta.view(-1)).sum(), Y_in, retain_graph=True)[0]

        Y += step_size * float(0.5) * derivative + np.sqrt(step_size) * U
        if j >= burn_in:
            if Y_samples is not None:
                Y_samples = torch.cat((Y_samples, Y), 0)
            else:
                Y_samples = Y
            Y_in.detach()
    return Y_samples.mean(0).unsqueeze(0)


def langevin_sample(Encoder, steps=100, step_size=0.001, sigma=1, initial=None, burn_in=90, batch=1):
    # initialize
    # noise = lambda x: x + sigma * torch.randn_like(x)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if initial is not None:
        Y = initial.to(device)
    else:
        Y = torch.rand([batch, 3, 32, 32]).to(device)

    # Langevin updates
    Y_samples = None
    for j in range(steps):
        U = sigma * torch.randn_like(Y)
        # U = 0                               # no noise
        Y_in = Variable(Y, requires_grad=True)
        derivative = torch.autograd.grad(Encoder(Y_in).sum(), Y_in, retain_graph=True)[0]

        Y += step_size * float(0.5) * derivative + np.sqrt(step_size) * U
        if j >= burn_in:
            if Y_samples is not None:
                Y_samples = torch.cat((Y_samples, Y), 0)
            else:
                Y_samples = Y
            Y_in.detach()
    return Y_samples.mean(0).unsqueeze(0)


class onelayer(nn.Module):

    # Our batch shape for input x is (3, 32, 32)
    # this model has 270 parameters / a FRAME model

    def __init__(self):
        super(onelayer, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3, stride=1, bias=False)
        self.softplus = nn.Softplus()
    def forward(self, x):
        x = (self.conv1(x))
        x = x.sum((2, 3))

        return x





class param_generator(torch.nn.Module):

    # Our batch shape for input x is (3, 32, 32)

    def __init__(self):
        super(param_generator, self).__init__()
        self.relu = nn.LeakyReLU(0.5, True)
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(3, 75, kernel_size=9, stride=3, dilation=2)

    def forward(self, x):
        x = self.sigmoid(self.conv1(x))
        x = x.view(-1)
        return x



class param_learner(torch.nn.Module):

    # Our batch shape for input x is (3, 32, 32)

    def __init__(self):
        super(param_learner, self).__init__()
        self.relu = nn.LeakyReLU(0.5, True)
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(3, 75, kernel_size=9, stride=3, dilation=2)

    def forward(self, x):
        x = self.sigmoid(self.conv1(x))
        x = x.view(-1)
        return x



class coef_learner(torch.nn.Module):

    # Our batch shape for input x is (3, 32, 32)

    def __init__(self):
        super(coef_learner, self).__init__()
        self.relu = nn.LeakyReLU(0.5, True)
        self.sigmoid = nn.Sigmoid()
        self.fc1 = torch.nn.Linear(3 * 32 * 32, 900)
        self.fc2 = torch.nn.Linear(900, 100)
        self.fc3 = torch.nn.Linear(100, 10)
        self.fc = torch.nn.Linear(3 * 32 * 32, 10)

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        # x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        # x = self.relu(self.fc3(x))
        x = self.relu(self.fc(x))
        return x




