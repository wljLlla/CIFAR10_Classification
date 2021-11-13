import time
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

from net import *

import argparse
import sys
import traceback
best_acc = 0
def evalTest(testloader, model, gpu = True):
    model.eval()
    testIter = iter(testloader)
    acc = 0.0
    NBatch = 0
    for i, data in enumerate(testIter, 0):
        NBatch += 1
        batchX, batchY = data
        if gpu:
            batchX = batchX.cuda()
            batchY = batchY.cuda()
        logits = model.forward(batchX)

        if gpu:
            pred = np.argmax(logits.cpu().detach().numpy(), axis = 1)
            groundTruth = batchY.cpu().detach().numpy()
        else:
            pred = np.argmax(logits.detach().numpy(), axis = 1)
            groundTruth = batchY.detach().numpy()
        acc += np.mean(pred == groundTruth)
    accTest = acc / NBatch
    print("Test accuracy: " + str(accTest))
    return accTest

def train(DATASET = 'CIFAR10', network = 'CIFAR10CNN', NEpochs = 200, imageWidth = 32,
          imageHeight = 32, imageSize = 32*32, NChannels = 3, NClasses = 10,
          BatchSize = 32, learningRate = 1e-3, NDecreaseLR = 20, eps = 1e-3,
          AMSGrad = True, model_dir = "checkpoints/CIFAR10/", model_name = "ckpt.pth", gpu = True, tt = 0):

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    print('DATASET:' + DATASET)

    mu = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    sigma = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    Normalize = transforms.Normalize(mu.tolist(), sigma.tolist())
    tsf = {
        'train': transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees = 10, translate = [0.1, 0.1], scale = [0.9, 1.1]),
                transforms.ToTensor(),
                Normalize
            ]),
        'test': transforms.Compose(
            [
                transforms.ToTensor(),
                Normalize
            ])
    }

    trainset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train = True,
                                            download=True, transform = tsf['train'])
    testset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train = False,
                                           download=True, transform = tsf['test'])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size = BatchSize,
                                              shuffle = True, num_workers = 1)
    testloader = torch.utils.data.DataLoader(testset, batch_size = 1000,
                                         shuffle = False, num_workers = 1)

    trainIter = iter(trainloader)
    testIter = iter(testloader)

    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)

    model = CIFAR10CNN(NChannels).to(DEVICE)
    # model = CIFAR10CNN(NChannels)
    # model = torch.nn.DataParallel(model, device_ids=[0, 1, 3])

    criterion.to(DEVICE)
    softmax.to(DEVICE)

    print(model)

    optimizer = optim.Adam(params = model.parameters(), lr = learningRate, eps = eps, amsgrad = AMSGrad)
    NBatch = len(trainset) / BatchSize
    # cudnn.benchmark = True
    model.train()

    w1_list = []
    mse1_list = []
    acc1_list = []

    for epoch in range(NEpochs):
        lossTrain = 0.0
        accTrain = 0.0
        # model.train()
        # trainIter = iter(trainloader)
        for batch_idx, (data, target) in enumerate(trainloader):
            data, target = data.to(DEVICE), target.to(DEVICE)

            output = model(data)
            prob = softmax(output)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossTrain += loss.cpu().detach().numpy() / NBatch

            if gpu:
                pred = np.argmax(prob.cpu().detach().numpy(), axis = 1)
                groundTruth = target.cpu().detach().numpy()
            else:
                pred = np.argmax(prob.detach().numpy(), axis = 1)
                groundTruth = target.detach().numpy()

            acc = np.mean(pred == groundTruth)
            accTrain += acc / NBatch

        if (epoch + 1) % NDecreaseLR == 0:
            learningRate = learningRate / 2.0
            for param_group in optimizer.param_groups:
                param_group['lr'] = learningRate

        w1_list.append(epoch)
        mse1_list.append(lossTrain)
        acc1_list.append(accTrain)

        # w_list.append(w1_list)
        # mse_list.append(mse1_list)
        # acc_list.append(acc1_list)

        print("Epoch: " + str(epoch) + "  Loss: " + str(lossTrain) + "  Train accuracy: " + str(accTrain))

        accTest = evalTest(testloader, model, gpu = gpu)

    w_list.append(w1_list)
    mse_list.append(mse1_list)
    acc_list.append(acc1_list)

    # print(w_list)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    accFinale = evalTest(testloader, model, gpu = gpu)

    global best_acc
    print(best_acc)
    if accFinale > best_acc:

        tune = tt
        best_acc = accFinale
        print('tune = ' + str(tune))
        torch.save(model, model_dir + model_name)
        print("Model saved")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type = str, default = 'CIFAR10')
    parser.add_argument('--network', type = str, default = 'CIFAR10CNN')
    parser.add_argument('--epochs', type = int, default = 100)
    parser.add_argument('--eps', type = float, default = 1e-3)
    parser.add_argument('--AMSGrad', type = bool, default = True)
    # parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--batch_size', type = int, default = 0)
    parser.add_argument('--learning_rate', type = float, default = 1e-3)
    parser.add_argument('--decrease_LR', type = int, default = 20)

    parser.add_argument('--nogpu', dest='gpu', action='store_false')
    parser.set_defaults(gpu=True)
    args = parser.parse_args()

    model_dir = "checkpoints/" + args.dataset + '/'
    model_name = "ckpt.pth"

    imageWidth = 32
    imageHeight = 32
    imageSize = imageWidth * imageHeight
    NChannels = 3
    NClasses = 10
    network = 'CIFAR10CNN'

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    # device_ids = [0, 1, 3]


    w_list = []
    mse_list = []
    acc_list = []

    tune = 0

    for i in range(8):
        print('now tunes: '+str(i))
        args.batch_size += 100
        train(DATASET = args.dataset, network = network, NEpochs = args.epochs, imageWidth = int(imageWidth),
              imageHeight = imageHeight, imageSize = imageSize, NChannels = NChannels, NClasses = NClasses,
              BatchSize = args.batch_size, learningRate = args.learning_rate, NDecreaseLR = args.decrease_LR, eps = args.eps,
              AMSGrad = args.AMSGrad, model_dir = model_dir, model_name = model_name, gpu = args.gpu, tt = i)

    print('the best model is model'+str(tune))
    plt.title('Loss')
    for i in range(8):
        plt.plot(w_list[i], mse_list[i], label='BatchSize = :'+str(100+i*100))

    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(fontsize=8)
    plt.show()

    # print(w_list[0])

    plt.title('Accuracy')
    for i in range(8):
        plt.plot(w_list[i], acc_list[i], label='BatchSize = :'+str(100+i*100))

    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(fontsize=8)
    plt.show()

    # plt.title('BatchSize  = ' + str(args.batch_size) + '    Loss')
    # plt.plot(w_list, mse_list)
    # plt.ylabel('Loss')
    # plt.xlabel('epoch')
    # plt.show()
    #
    # plt.title('BatchSize = ' + str(args.batch_size) + '    Accuracy')
    # plt.plot(w_list, acc_list)
    # plt.ylabel('Accuracy')
    # plt.xlabel('epoch')
    # plt.show()