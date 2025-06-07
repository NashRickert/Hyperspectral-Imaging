import sys
import os
import torch
import pickle
import numpy as np
import pandas as pd
import scipy.io as sio
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

from abc import ABC

import torch.nn as nn
from torch import reshape

import random
import torch.optim as optim
from torchsummary import summary

from kan import *
torch.set_default_dtype(torch.float64)

np.random.seed(7)  # Initialize seed to get reproducible results
random.seed(7)
torch.manual_seed(7)
torch.cuda.manual_seed(7)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def loadata(name):
    data_path = os.path.join(os.getcwd(), '')
    if name == 'IP':
        dat = sio.loadmat('../Data/Indian_pines_corrected.mat', verify_compressed_data_integrity=False)['indian_pines_corrected']
        label = sio.loadmat(os.path.join(data_path, '../Data/Indian_pines_gt.mat'), verify_compressed_data_integrity=False)['indian_pines_gt']
        return dat, label
    elif name == 'SA':
        dat = sio.loadmat(os.path.join(data_path, '../Data/Salinas_corrected.mat'))['salinas_corrected']
        label = sio.loadmat(os.path.join(data_path, '../Data/Salinas_gt.mat'))['salinas_gt']
        return dat, label


def padWithZeros(Xc, margin=2):
    newX = np.zeros((Xc.shape[0] + 2 * margin, Xc.shape[1] + 2 * margin, Xc.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:Xc.shape[0] + x_offset, y_offset:Xc.shape[1] + y_offset, :] = Xc
    return newX


def createImageCubes(Xc, yc, window=5, removeZeroLabels=True):
    margin = int((window - 1) / 2)
    zeroPaddedX = padWithZeros(Xc, margin=margin)
    # split patches
    patchesData = np.zeros((Xc.shape[0] * Xc.shape[1], window, window, Xc.shape[2]))
    patchesLabels = np.zeros((Xc.shape[0] * Xc.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = yc[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
    return patchesData, patchesLabels

# Load and pre-process the data
data = 'IP'  # 'IP': Indian Pines, 'SA': Salinas 
trainx, train_y = loadata(data)
trainx, train_y = createImageCubes(trainx, train_y, window=1)

# Reshape as a 4-D TENSOR
trainx = np.reshape(trainx, (trainx.shape[0], trainx.shape[1], trainx.shape[2],
                             trainx.shape[3], 1))

# Shuffle dataset and reduce dataset size
ind = [i for i in range(trainx.shape[0])]
np.random.shuffle(ind)
trainx = trainx[ind][:, :, :, :, :]
train_y = train_y[ind][:]

# Transpose dimensions to fit Pytorch order
trainx = trainx.transpose((0, 4, 3, 1, 2))

# Separate 50% of the dataset for training
train_ind, val_ind = train_test_split(range(len(trainx)), test_size=0.50, random_state=7)
trainX = np.array(trainx[train_ind])
trainY = np.array(train_y[train_ind])
valX = np.array(trainx[val_ind])
valY = np.array(train_y[val_ind])


def get_class_distributionIP(train_y):
    """Get number of samples per class"""
    count_dict = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0, "10": 0, "11": 0,
                  "12": 0, "13": 0, "14": 0, "15": 0}
    for i in train_y:
        if i == 0:
            count_dict['0'] += 1
        elif i == 1:
            count_dict['1'] += 1
        elif i == 2:
            count_dict['2'] += 1
        if i == 3:
            count_dict['3'] += 1
        elif i == 4:
            count_dict['4'] += 1
        elif i == 5:
            count_dict['5'] += 1
        if i == 6:
            count_dict['6'] += 1
        elif i == 7:
            count_dict['7'] += 1
        elif i == 8:
            count_dict['8'] += 1
        if i == 9:
            count_dict['9'] += 1
        elif i == 10:
            count_dict['10'] += 1
        elif i == 11:
            count_dict['11'] += 1
        if i == 12:
            count_dict['12'] += 1
        elif i == 13:
            count_dict['13'] += 1
        elif i == 14:
            count_dict['14'] += 1
        elif i == 15:
            count_dict['15'] += 1

    return count_dict


def select(train_x, indexes):
    temp = np.zeros((train_x.shape[0], 1, len(indexes), train_x.shape[3], train_x.shape[4]))
    for nb in range(0, len(indexes)):
        temp[:, :, nb, :, :] = train_x[:, :, indexes[nb], :, :]
    train_x = temp.astype(np.float32)
    return train_x

def normalize(train_x):
    """Normalize and returns the calculated means and stds for each band"""
    trainxn = train_x.copy()
    dim = trainxn.shape[2]
    means = np.zeros((dim, 1))
    stds = np.zeros((dim, 1))
    for n in range(dim): # Apply normalization to the data that is already in Pytorch format
        means[n, ] = np.mean(trainxn[:, :, n, :, :])
        stds[n, ] = np.std(trainxn[:, :, n, :, :])
        trainxn[:, :, n, :, :] = (trainxn[:, :, n, :, :] - means[n, ]) / (stds[n, ])
    return trainxn, means, stds


def applynormalize(testx, means, stds):
    """Apply normalization based on previous calculated means and stds"""
    testxn = testx.copy()
    for n in range(testx.shape[2]):
        testxn[:, :, n, :, :] = (testxn[:, :, n, :, :] - means[n, ]) / (stds[n, ])
    return testxn

# Normalize using the training set
trainX, means, stds = normalize(trainX)
# Apply the same normalization to the validation set
valX = applynormalize(valX, means, stds)

# Make tensors into inputs for KAN
trainX =np.squeeze(trainX)
valX = np.squeeze(valX)
trainY =np.squeeze(trainY)
valY =np.squeeze(valY)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
print(device)

dataset = {}
dtype = torch.get_default_dtype()
dataset['train_input'] = torch.from_numpy(trainX).type(dtype).to(device)
dataset['test_input'] = torch.from_numpy(valX).type(dtype).to(device)
dataset['train_label'] = torch.from_numpy(trainY).type(torch.long).to(device)
dataset['test_label'] =torch.from_numpy(valY).type(torch.long).to(device)

# model = KAN(width=[200,32,32,32,16], grid = 1, k=5, seed=42, device=device, symbolic_enabled=False)

model = KAN(width=[200,32,32,32,16], grid = 10, k=3, seed=42, device=device, symbolic_enabled=False)

def train_acc():
    return torch.mean((torch.argmax(model(dataset['train_input']), dim=1) == dataset['train_label']).type(dtype))

def test_acc():
    return torch.mean((torch.argmax(model(dataset['test_input']), dim=1) == dataset['test_label']).type(dtype))


train = true
if train:
    results = model.fit(dataset, batch=1024, opt="LBFGS", steps=10, lamb=1e-4, lamb_entropy = 2e-01, metrics=(train_acc, test_acc), loss_fn=torch.nn.CrossEntropyLoss());
    print("Training Accuracy:",results['train_acc'][-1])
    print("Test Accuracy:", results['test_acc'][-1])
else:
    model.load_state_dict(torch.load('temp_model.pt'))

log_file = open("KAN_SD1.log", "w")
sys.stdout = log_file
sd = model.state_dict()
for name, tensor in sd.items():
    print(name)
    print(tensor.shape)
    print(tensor)
log_file.close()

# print(len(dataset['train_input']))
# print(dataset['train_input'])
# print(len(dataset['train_label']))
# print(dataset['train_label'])
# print(len(dataset['test_input']))
# print(dataset['test_input'])
# print(len(dataset['test_label']))
# print(dataset['test_label'])


# torch.save(model.state_dict(), 'temp_model.pt')
