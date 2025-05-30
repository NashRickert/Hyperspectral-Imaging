"""
Dirk Kaiser - Montana State University
Convolutional Neural Network Reference Model

Adapted from Giorgio Morales' HSI-BandSelection: https://github.com/NISL-MSU/HSI-BandSelection
This code was originally created to classify based on a smaller selection of hyperspectral bands in the dataset 
but was addapted to classify using all availible hyperspectral bands excluding bands associated with water in the Indian Pines dataset (200).
To run this code, install all dependencies and run in python.
Results using model produce a classification accuracy of 99.5317% with 200 bands
This script will by default use the corrected Inidian Pines dataset mat file from Giorgio's work (https://github.com/NISL-MSU/HSI-BandSelection/tree/master/src/HSIBandSelection/Data),
which is the uses the hyperspectral images orignally collected here https://purr.purdue.edu/publications/1947/1
"""

import os
import random
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
    else:
        raise ValueError()

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
trainx, train_y = createImageCubes(trainx, train_y, window=5)

# Reshape as a 4-D TENSOR
trainx = np.reshape(trainx, (trainx.shape[0], trainx.shape[1], trainx.shape[2],
                             trainx.shape[3], 1))

# Shuffle dataset and reduce dataset size
np.random.seed(seed=7)  # Initialize seed to get reproducible results
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

from abc import ABC

import torch.nn as nn
from torch import reshape


def weight_reset(m):
    """Reset model weights"""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        m.reset_parameters()


class Hyper3DNetLite(nn.Module, ABC):
    def __init__(self, img_shape=(1, 50, 25, 25), classes=2, data='Kochia'):
        super(Hyper3DNetLite, self).__init__()
        if data == 'Kochia' or data == 'Avocado':
            stride = 2
            out = 7
        else:
            stride = 1
            out = 5
        self.classes = classes
        self.img_shape = img_shape

        self.conv_layer1 = nn.Sequential(nn.Conv3d(in_channels=img_shape[0], out_channels=16, kernel_size=3, padding=1),
                                         nn.ReLU(), nn.BatchNorm3d(16))
        self.conv_layer2 = nn.Sequential(nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
                                         nn.ReLU(), nn.BatchNorm3d(16))
        self.sepconv1 = nn.Sequential(nn.Conv2d(in_channels=16 * img_shape[1], out_channels=16 * img_shape[1],
                                                kernel_size=5, padding=2, groups=16 * img_shape[1]), nn.ReLU(),
                                      nn.Conv2d(in_channels=16 * img_shape[1], out_channels=320,
                                                kernel_size=1, padding=0), nn.ReLU(), nn.BatchNorm2d(320))
        self.sepconv2 = nn.Sequential(nn.Conv2d(in_channels=320, out_channels=320,
                                                kernel_size=3, padding=1, stride=stride, groups=320), nn.ReLU(),
                                      nn.Conv2d(in_channels=320, out_channels=256,
                                                kernel_size=1, padding=0), nn.ReLU(), nn.BatchNorm2d(256))
        self.sepconv3 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256,
                                                kernel_size=3, padding=1, stride=stride, groups=256), nn.ReLU(),
                                      nn.Conv2d(in_channels=256, out_channels=256,
                                                kernel_size=1, padding=0), nn.ReLU(), nn.BatchNorm2d(256))
        self.average = nn.AvgPool2d(kernel_size=out)

        if classes == 2:
            self.fc1 = nn.Linear(256, 1)
        else:
            self.fc1 = nn.Linear(256, self.classes)

    def forward(self, x):
        # This is 128, 1, 200, 5, 5
        # print("Data shape in forward: ", x.shape)

        # 3D Feature extractor
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        # Reshape 3D-2D
        x = reshape(x, (x.shape[0], self.img_shape[1] * 16, self.img_shape[2], self.img_shape[3]))
        # 2D Spatial encoder
        x = self.sepconv1(x)
        x = self.sepconv2(x)
        x = self.sepconv3(x)
        # Global Average Pooling
        x = self.average(x)
        x = reshape(x, (x.shape[0], x.shape[1]))
        if self.classes == 2:
            x = self.fc1(x)
        else:
            x = self.fc1(x)
        return x

import random
import torch.optim as optim
from torchsummary import summary
np.random.seed(seed=7)  # Initialize seed to get reproducible results
random.seed(7)
torch.manual_seed(7)
torch.cuda.manual_seed(7)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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


class CNNObject:
    """Helper class used to store the main information of a CNN for training"""

    def __init__(self, model, criterion, optimizer):
        self.network = model
        self.criterion = criterion
        self.optimizer = optimizer


class CNNTrainer():

    def __init__(self):
        self.nbands = None
        self.model = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.classes = None

    def defineModel(self, nbands, windowSize, train_y):
        """Model declaration method"""
        self.classes = len(np.unique(train_y))
        model = Hyper3DNetLite(img_shape=(1, nbands, windowSize, windowSize), classes=int(self.classes), data=data)
        model.to(self.device)
        # Training parameters
        class_count = [i for i in get_class_distributionIP(train_y).values()]
        class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))

        optimizer = optim.Adadelta(model.parameters(), lr=1.0)

        self.nbands = nbands

        self.model = CNNObject(model, criterion, optimizer)

    def trainFold(self, trainx, trainy, batch_size, 
                          epochs, valx, valy, filepath, printProcess=False):
        np.random.seed(seed=7)  # Initialize seed to get reproducible results (doesn't seem to work in Colab)
        random.seed(7)
        torch.manual_seed(7)
        torch.cuda.manual_seed(7)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
      
        print("Training model.....")
        # Prints summary of the modelif printProcess:      
        if printProcess:
            summary(self.model.network, (1, trainx.shape[2], trainx.shape[3], trainx.shape[4]))
            
        indexes = np.arange(len(trainx))  # Prepare list of indexes for shuffling
        T = np.ceil(1.0 * len(trainx) / batch_size).astype(np.int32)  # Compute the number of steps in an epoch
        val_acc = 0
        loss = 1
        for epoch in range(epochs):  # Epoch loop
            # Shuffle indexes when epoch begins
            print("(Temporary) Epoch: ", epoch)

            self.model.network.train()  # Sets training mode
            running_loss = 0.0
            for step in range(T):  # Batch loop
                # Generate indexes of the batch
                inds = indexes[step * batch_size:(step + 1) * batch_size]

                # Get actual batches
                trainxb = torch.from_numpy(trainx[inds]).float().to(self.device)
                trainyb = torch.from_numpy(trainy[inds]).long().to(self.device)
                # print("Shape of input data is ", trainxb.shape)

                # zero the parameter gradients
                self.model.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model.network(trainxb)
                loss = self.model.criterion(outputs, trainyb)
                loss.backward()
                self.model.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if step % 10 == 9 and printProcess:  # print every 10 mini-batches
                    print('[%d, %5d] loss: %.5f' %
                          (epoch + 1, step + 1, running_loss / 10))
                    running_loss = 0.0

            # Validation step
            ytest, ypred = self.evaluateFold(valx, valy, batch_size)
            correct_pred = (np.array(ypred) == ytest).astype(float)
            oa = correct_pred.sum() / len(correct_pred) * 100  # Calculate accuracy

            # Save model if accuracy improves
            if oa >= val_acc:
                val_acc = oa
                torch.save(self.model.network.state_dict(), filepath)  # saves checkpoint

            if printProcess:
                print('VALIDATION: Epoch %d, loss: %.5f, acc: %.3f, best_acc: %.3f' %
                      (epoch + 1, loss.item(), oa.item(), val_acc))

    def evaluateFold(self, valx, valy, batch_size):
        ypred = []
        with torch.no_grad():
            self.model.network.eval()
            Teva = np.ceil(1.0 * len(valx) / batch_size).astype(np.int32)
            indtest = np.arange(len(valx))
            for b in range(Teva):
                inds = indtest[b * batch_size:(b + 1) * batch_size]
                ypred_batch = self.model.network(torch.from_numpy(valx[inds]).float().to(self.device))
                y_pred_softmax = torch.log_softmax(ypred_batch, dim=1)
                _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
                ypred = ypred + (y_pred_tags.cpu().numpy()).tolist()
        ytest = torch.from_numpy(valy).long().cpu().numpy()

        return ytest, ypred

    def loadModel(self, path):
        self.model.network.load_state_dict(torch.load(path))

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

# Initialize model and train (USE GPU!: Runtime -> Change runtime type)
model = CNNTrainer()
model.defineModel(nbands=200, windowSize=5, train_y=trainY)
model.trainFold(trainx=trainX, trainy=trainY, valx=valX, valy=valY, 
                        batch_size=128, epochs=50, filepath="temp_model", printProcess=False)  # Set printProcess=True to see the training process 

# Validate
model.loadModel("temp_model")
ytest, ypred = model.evaluateFold(valx=valX, valy=valY, batch_size=128)
correct_pred = (np.array(ypred) == ytest).astype(float)
oa = correct_pred.sum() / len(correct_pred) * 100
prec, rec, f1, support = precision_recall_fscore_support(ytest, ypred, average='macro')

print("Accuracy = " + str(oa))
print("Precision = " + str(prec))
print("Recall = " + str(rec))
print("F1 = " + str(f1))

