#this has been taken from Paolos link

import glob
# TESTING
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import roc_curve, confusion_matrix
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don’t have any parameters
from sklearn.metrics import accuracy_score


class Net(nn.Module):
    num_classes = 1
    def __init__(self,  num_classes):
        super(Net, self).__init__()
        self.bn0 = nn.BatchNorm1d(54)
        self.conv1 = nn.Conv1d(in_channels=54, out_channels=100, kernel_size=3, stride=2, padding=1)
        torch.nn.init.kaiming_uniform_(self.conv1.weight)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1_bn = nn.BatchNorm1d(100)

        self.conv2 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3, stride=2, padding=1)
        torch.nn.init.kaiming_uniform_(self.conv2.weight)
        self.conv2_bn = nn.BatchNorm1d(100)

        self.fc1 = nn.Linear(2600, num_classes)
        torch.nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, x):
        x = self.bn0(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.conv1_bn(x)

        x = self.pool(F.relu(self.conv2(x)))
        x = self.conv2_bn(x)

        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc1(x))

        return x


class Net_th(nn.Module):
    def __init__(self,  num_classes, n_features):
        super(Net_th, self).__init__()
        self.bn0 = nn.BatchNorm1d(n_features)
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=100, kernel_size=3, stride=2, padding=1)
        torch.nn.init.kaiming_uniform_(self.conv1.weight)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1_bn = nn.BatchNorm1d(100)

        self.conv2 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3, stride=2, padding=1)
        torch.nn.init.kaiming_uniform_(self.conv2.weight)
        self.conv2_bn = nn.BatchNorm1d(100)

        self.rnn = nn.LSTM(input_size=100,hidden_size=26,num_layers=3, dropout=0.1, batch_first=True, bidirectional = True)
        self.drop = nn.Dropout(p = 0.1)

        self.fc1 = nn.Linear(26*2, num_classes)
        torch.nn.init.xavier_uniform_(self.fc1.weight)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.bn0(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.conv1_bn(x)
        x = self.drop(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.conv2_bn(x)
        x = self.drop(x)
        x = x.transpose_(2, 1)
        x, (h, c) = self.rnn(x)
        cat = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)
        cat = self.drop(cat)
        x = self.fc1(cat)
        return x

class Net_project(nn.Module):
    def __init__(self,  num_classes, n_features, numHN, numFilter,  dropOutRate):
        super(Net_th, self).__init__()
        self.bn0 = nn.BatchNorm1d(n_features)
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=numFilter, kernel_size=3, stride=2, padding=1)
        torch.nn.init.kaiming_uniform_(self.conv1.weight)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1_bn = nn.BatchNorm1d(100)

        self.conv2 = nn.Conv1d(in_channels=numFilter, out_channels=numFilter, kernel_size=3, stride=2, padding=1)
        torch.nn.init.kaiming_uniform_(self.conv2.weight)
        self.conv2_bn = nn.BatchNorm1d(100)

        self.rnn = nn.LSTM(input_size=numFilter,hidden_size=numHN,num_layers=3, dropout=dropOutRate, batch_first=True, bidirectional = True)
        self.drop = nn.Dropout(p = dropOutRate)

        self.fc1 = nn.Linear(numHN*2, num_classes)
        torch.nn.init.xavier_uniform_(self.fc1.weight)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.bn0(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.conv1_bn(x)
        x = self.drop(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.conv2_bn(x)
        x = self.drop(x)
        x = x.transpose_(2, 1)
        x, (h, c) = self.rnn(x)
        cat = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)
        cat = self.drop(cat)
        x = self.fc1(cat)
        return x