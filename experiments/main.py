import torch
import numpy as np

torch.manual_seed(17)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(17)
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_value_, clip_grad_norm_
from collections import OrderedDict

import os
from pau.utils import activationfunc

font = {'family': 'normal',
        'weight': 'bold',
        'size': 22}

writer = None
matplotlib.rc('font', **font)

cnt = 0

activation_functions = dict({
    "pade_optimized_leakyrelu_abs": "pade_optimized_leakyrelu_abs",
    #"leakyrelu": 'leakyrelu',
    #"relu": 'relu',
    #"swish": 'swish',
    #"sigmoid": 'sigmoid',
    #"tanh": 'tanh',
    #"prelu": 'prelu',
    #"relu6": 'relu6'
})


def vgg_block(num_convs, in_channels, num_channels, actv_function):
    layers = []
    for i in range(num_convs):
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=num_channels, kernel_size=3, padding=1)]
        in_channels = num_channels
    layers += [actv_function()]
    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, activation_func):
        super(VGG, self).__init__()
        actv_function = activationfunc(activation_func).get_activationfunc

        self.conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
        layers = []
        for (num_convs, in_channels, num_channels) in self.conv_arch:
            layers += [vgg_block(num_convs, in_channels, num_channels, actv_function)]
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


class LeNet5(nn.Module):
    def __init__(self, activation_func):
        super(LeNet5, self).__init__()
        actv_function = activationfunc(activation_func).get_activationfunc

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('actv1', actv_function()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('actv3', actv_function()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('actv5', actv_function())
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120, 84)),
            ('actv6', actv_function()),
            ('f7', nn.Linear(84, 10)),
            ('sig7', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output


class CONV(nn.Module):
    def __init__(self, activation_func):
        super(CONV, self).__init__()
        actv_function = activationfunc(activation_func).get_activationfunc

        self.conv1 = nn.Conv2d(1, 128, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(128, affine=True)
        self.drop1 = nn.Dropout2d(0.2)

        self.conv2 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(128, affine=True)
        self.drop2 = nn.Dropout2d(0.2)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128, affine=True)
        self.drop3 = nn.Dropout2d(0.2)

        self.fc1 = nn.Linear(128, 10)

        self.actv1 = actv_function()
        self.actv2 = actv_function()
        self.actv3 = actv_function()

        self.activations = [self.actv1, self.actv2, self.actv3]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activations[0](x)
        x = self.drop1(x)
        x = F.max_pool2d(x, 3, 2)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activations[1](x)
        x = self.drop2(x)
        x = F.max_pool2d(x, 3, 2)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activations[2](x)
        x = self.drop3(x)
        x = F.avg_pool2d(x, 6, 1)

        x = x.view(-1, 128)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


def train(model, device, train_loader, optimizer, optimizer_activation, epoch):
    global cnt
    model.train()

    running_loss = 0.
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        if optimizer_activation is not None:
            optimizer_activation.zero_grad()

        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        clip_grad_norm_(model.parameters(), 5., norm_type=2)

        optimizer.step()
        if optimizer_activation is not None:
            optimizer_activation.step()

        # collect statistics
        running_loss += loss.item()

    writer.add_scalar('train/loss_epoch', running_loss / len(train_loader.dataset), epoch)


def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    writer.add_scalar('test/loss', test_loss, epoch)
    writer.add_scalar('test/accuracy', acc, epoch)

    print('\nTest set: Epoch: {}, Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(epoch, test_loss,
                                                                                            correct,
                                                                                            len(test_loader.dataset),
                                                                                            acc))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=17, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='dataset to use')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--arch', type=str, required=True, help='vgg, lenet or conv')
    parser.add_argument('--optimizer', type=str, default="Adam")
    args = parser.parse_args()

    networks = dict({
        "vgg": VGG,
        "lenet": LeNet5,
        "conv": CONV
    })

    network = networks[args.arch]

    global writer
    global cnt
    for activation_function_key in activation_functions.keys():
        print("---" * 42)
        print("Starting with dataset: {}, activation function: {}".format(args.dataset, activation_function_key))
        print("---" * 42)
        # writer = SummaryWriter(comment=activation_function_key)
        save_path = 'experiments/{}_{}_{}_seed{}/'.format(args.dataset, args.arch, args.optimizer,
                                                             args.seed) + activation_function_key
        writer = SummaryWriter(save_path)

        writer.add_scalar('configuration/batch size', args.batch_size)
        writer.add_scalar('configuration/learning rate', args.lr)
        writer.add_scalar('configuration/seed', args.seed)

        cnt = 0

        use_cuda = not args.no_cuda and torch.cuda.is_available()

        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)

        device = torch.device("cuda" if use_cuda else "cpu")

        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        if args.dataset == 'mnist':
            train_loader = torch.utils.data.DataLoader(
                datasets.MNIST('../data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize((32, 32)),
                                   transforms.RandomRotation(30),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ])),
                batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
            test_loader = torch.utils.data.DataLoader(
                datasets.MNIST('../data', train=False, transform=transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])),
                batch_size=args.test_batch_size, shuffle=True, **kwargs)

        elif args.dataset == 'fmnist':
            train_loader = torch.utils.data.DataLoader(
                datasets.FashionMNIST('../data', train=True, download=True,
                                      transform=transforms.Compose([
                                          transforms.Resize((32, 32)),
                                          transforms.RandomRotation(30),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))
                                      ])),
                batch_size=args.batch_size, shuffle=True, **kwargs)
            test_loader = torch.utils.data.DataLoader(
                datasets.FashionMNIST('../data', train=False, transform=transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])),
                batch_size=args.test_batch_size, shuffle=True, **kwargs)
        else:
            raise ValueError('dataset error')

        model = network(activation_func=activation_function_key).to(device)

        params = list()
        params_activation = list()
        for p in model.named_parameters():
            if 'weight_center' in p[0] or 'weight_numerator' in p[0] or 'weight_denominator' in p[0]:
                if p[1].requires_grad:
                    params_activation.append(p[1])
            else:
                params.append(p[1])
        if args.optimizer.lower() == "adam":
            optimizer = optim.Adam(params, lr=args.lr)
        else:
            optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum)

        if "pade" in activation_function_key:
            if args.optimizer.lower() == "adam":
                optimizer_activation = optim.Adam(params_activation, lr=args.lr)
            else:
                optimizer_activation = optim.SGD(params_activation, lr=args.lr, momentum=args.momentum)
        else:
            optimizer_activation = None

        # test initialization
        test(model, device, test_loader, 0)

        # run training
        for epoch in range(1, args.epochs + 1):
            train(model, device, train_loader, optimizer, optimizer_activation, epoch)
            test(model, device, test_loader, epoch)

        writer.close()

        # save trained model
        if args.save_model:
            torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))


if __name__ == '__main__':
    main()
