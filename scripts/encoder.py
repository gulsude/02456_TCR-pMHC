import torch
import os
import sys
import random
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import date
from sklearn.metrics import matthews_corrcoef
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc

#-------- Import Modules from project--------#
import encoding as enc
from model import Net, Net_thesis, Net_project
import functions as func

#-------- Set Device --------#

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
else:
    print('No GPUs available. Using CPU instead.')
    device = torch.device('cpu')

#-------- Seeds --------#

seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

torch.use_deterministic_algorithms(True)

#-------- Directories --------#

DATADIR = '/data/'
TRAINDIR = '../data/train'
VALIDATIONDIR = '../data/validation'
MATRICES = '/data/Matrices'



# -------- Unzip Train --------#

try:
    if len(os.listdir(TRAINDIR)) != 0:
        print("{} already unzipped.".format(TRAINDIR))
except:
    pass #!unzip.. / data / train.zip - d.. / data / train

# -------- Unzip Validation --------#


try:
    if len(os.listdir(VALIDATIONDIR)) != 0:
        print("{} already unzipped.".format(VALIDATIONDIR))
except:
    pass #!unzip.. / data / validation.zip - d.. / data / validation

print('Train directory:\n\n', '\n'.join(str(p) for p in os.listdir(TRAINDIR)), '\n\n')
print('Validation directory:\n\n', '\n'.join(str(p) for p in os.listdir(VALIDATIONDIR)))

data_list = []
target_list = []

import glob

for fp in glob.glob("../data/train/*input.npz"):
    data = np.load(fp)["arr_0"]
    targets = np.load(fp.replace("input", "labels"))["arr_0"]
    data_list.append(data)
    target_list.append(targets)

print(len(data_list))
print(len(target_list))

data_partitions = len(data_list)

CNN = False # ONLY CNN
CNN_RNN = True # CNN + RNN

# Hyperparameters to fine-tune
embedding = "esm-1b"
numHN=64
numFilter=100
dropOutRate=0.1
keep_energy=True

# embedding of data

# create directory to fetch/store embedded
embedding_dir = '../data/embeddedFiles/'
try:
    os.mkdir(embedding_dir)
except:
    pass

# try to fecth if already exist
try:
    infile = open(embedding_dir + 'dataset-{}'.format(embedding), 'rb')
    data_list_enc = pickle.load(infile)
    infile.close()
    if len(data_list_enc) != len(data_list):
        raise Exception("encoding needed")

# if no prior file, then embbed:
except:
    data_list_enc = []
    if embedding == "Baseline":
        data_list_enc = data_list

    elif embedding == "esm-1b":
        for dataset in data_list:
            x_enc = np.array(func.extract_sequences(dataset, merge=True))
            x_enc = [enc.esm_1b_peptide(seq, pooling=False) for seq in x_enc]
            data_list_enc.append(x_enc)

        # save
        outfile = open(embedding_dir + 'dataset-{}'.format(embedding), 'wb')
        pickle.dump(data_list_enc, outfile)
        outfile.close()

    elif embedding == "esm_ASM":
        for dataset in data_list:
            x_enc = np.array(func.extract_sequences(dataset, merge=True))  # .values.tolist()
            x_enc = [enc.esm_ASM(seq, pooling=False) for seq in x_enc]
            data_list_enc.append(x_enc)

        # save
        outfile = open(embedding_dir + 'dataset-{}'.format(embedding), 'wb')
        pickle.dump(data_list_enc, outfile)
        outfile.close()

    else:
        for dataset in data_list:
            x_enc = func.extract_sequences(dataset, merge=True)
            x_enc = x_enc.tolist()
            x_enc = enc.encodePeptides(x_enc, scheme=embedding)
            data_list_enc.append(x_enc)

        # save
        outfile = open(embedding_dir + 'dataset-{}'.format(embedding), 'wb')
        pickle.dump(data_list_enc, outfile)
        outfile.close()
