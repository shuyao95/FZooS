'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import SubsetRandomSampler

import os
import argparse
import numpy as np

from attack.models import ResNet18
from split_data import *

# Training
def train(epoch, net, train_loader, device, optimizer, criterion, log_interval=10):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {:05d} [{:05d}/{:05d} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(train_loader.sampler.indices),
                100. * batch_idx / len(train_loader), loss.item()))
    


def test(epoch, net, test_loader, device, criterion):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    print('\nTest set: Average loss: {:.4f}, Accuracy: {:05d}/{:05d} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
    

def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--portion', type=float, default=1.0, metavar='P',
                        help='portion of classes (default: 1.0)')
    args = parser.parse_args()


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    path = os.path.join('./attack/ckpts', str(args.portion))
    if not os.path.exists(path):
        os.makedirs(path)

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

    train_data = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)

    test_data = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

    num_classes = len(classes)
    indices = np.arange(len(train_data))
    labels = np.asarray(train_data.targets)
    Xs, Ys = split_by_class(indices, labels, num_classes)
    indices_split, _ = generate_noniid_data(Xs, Ys, portion=args.portion, num_classes=num_classes)

    # Model
    print('==> Building model..')
    net = ResNet18()
    net = net.to(device)
    if device == 'cuda':
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    for i in range(num_classes):
        print('==> Train the model on client %02d..' %(i+1))
        
        train_indices = torch.from_numpy(indices_split[i])
        train_loader = torch.utils.data.DataLoader(
            train_data, sampler=SubsetRandomSampler(train_indices), batch_size=128, num_workers=2)
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=100, shuffle=False, num_workers=2)

        for epoch in range(start_epoch, start_epoch+100):
            train(epoch, net, train_loader, device, optimizer, criterion)
            test(epoch, net, test_loader, device, criterion)
            # scheduler.step()
            
        # Save checkpoint.
        torch.save(net.state_dict(), os.path.join(path, f"cifar10_{i}.pt"))
        
if __name__ == '__main__':
    main()