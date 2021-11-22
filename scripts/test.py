data_list = []
target_list = []
import torch
import numpy as np

import encoding as enc
import functions as f
"""
peptides = ["AGAG"]
embedded = enc.esm_1b(peptides, pooling=False)


#a = enc.esm_ASM(peptides, pooling=False)

print(embedded[0].shape)
print(embedded )
"""
# -------- Import Dataset --------#             #TO UPDATE: NO REPEAT P4 AND CONSERVE PARTITION FOR CROSS VAL

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
"""
X_train = np.concatenate(data_list[:-1])
y_train = np.concatenate(target_list[:-1])
nsamples, nx, ny = X_train.shape
print("Training set shape:", nsamples, nx, ny)
"""
X_val = np.concatenate(data_list[-1:])
y_val = np.concatenate(target_list[-1:])
X_val = X_val[:20]
nsamples, nx, ny = X_val.shape
print("Val set shape:", nsamples, nx, ny)
print(X_val)








#energy = f.extract_energy_terms(X_val)

