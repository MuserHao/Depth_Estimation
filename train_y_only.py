'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import os
import argparse
from energy_model import *
# from models import *
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

start_epoch = 0  # start from epoch 0 or last checkpoint epoch
batch_size = 32
# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')

net = onelayer()
net = net.to(device)
# beta_true = coef_learner()
# beta_true = beta_true.to(device)
# beta = coef_learner()
# beta = beta.to(device)

# net_true = onelayer()
# net_true = net_true.to(device)
# theta_true = param_generator()
# theta_true = theta_true.to(device)
# theta = param_learner()
# theta = theta.to(device)


# Set up true parameters for theta_true


np.random.seed(1)
torch.manual_seed(1)

# for param in beta_true.parameters():
#     param.data = (-4) * torch.rand_like(param.data, dtype=torch.float32, device=device) + 2
#     # print(param.data.shape)

# for param in net.parameters():
#     param.data = (-20) * torch.rand_like(param.data, dtype=torch.float32, device=device) + 10
#     # print(param.data)

# theta_true.conv1.weight.data = (-1 - 1) * torch.randn([75, 3, 9, 9], dtype=torch.float32, device=device) + 1
# theta_true.conv1.bias.data = (-1 - 1) * torch.randn([75], dtype=torch.float32, device=device) + 1


step_mc = 100
criterion = nn.MSELoss()
optimizer1 = optim.SGD(net.parameters(), lr=args.lr * 0.01, momentum=0.9, weight_decay=5e-4)
# optimizer2 = optim.SGD(beta.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train();
    # beta_true.eval()
    # beta.train()

    # theta.train()
    # net_true.eval();
    # theta_true.eval()

    train_loss = 0
    train_image_loss = 0

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs = Variable(inputs).to(device)
        optimizer1.zero_grad()
        # optimizer2.zero_grad()

        # outputs = theta(inputs)
        # targets = theta_true(inputs)


        # net.conv1.weight.data = targets.view(100, 3, 3, 3)

        Y_samples = langevin_sample(net, steps=step_mc, burn_in=step_mc - 10, batch=batch_size)  # ,
        #  initial=torch.rand([1, 3, 32, 32], dtype=torch.float32, device=device))
        # Y_true = langevin_sample(net, step_size=0.00001, steps=20000, burn_in=20000 - 10)

        # plt.imsave('Generated_img_{:05d}.png'.format(batch_idx), Y_true.squeeze(0).permute(1, 2, 0))

        # loss = -(net(Y_true).mean(0) - net(Y_samples).mean(0)).sum()

        loss = -(net(inputs) - net(Y_samples)).sum()
        loss.backward()
        optimizer1.step()
        # weight_loss = criterion(outputs, net.conv1.weight.data.view(-1))
        # weight_loss.backward()
        # optimizer2.step()
        image_loss = criterion(Y_samples, inputs.mean(0).unsqueeze(0))

        train_image_loss += image_loss.item()
        train_loss += loss.item()


        print('Loss: %.6f | Image loss: %.6f'
              % (train_loss / (batch_idx + 1), image_loss.item()))
        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Weight loss: %.3f'
        #     % (train_loss/(batch_idx+1), train_weight_loss/(batch_idx+1)))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch + 100):
    train(epoch)
