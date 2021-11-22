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

X_train = np.concatenate(data_list[:-1])
y_train = np.concatenate(target_list[:-1])
nsamples, nx, ny = X_train.shape
print("Training set shape:", nsamples, nx, ny)

X_val = np.concatenate(data_list[-1:])
y_val = np.concatenate(target_list[-1:])
nsamples, nx, ny = X_val.shape
print("Val set shape:", nsamples, nx, ny)

p_neg = len(y_train[y_train == 1]) / len(y_train) * 100
print("Percent positive samples in train:", p_neg)

p_pos = len(y_val[y_val == 1]) / len(y_val) * 100
print("Percent positive samples in val:", p_pos)

# make the data set into one dataset that can go into dataloader
train_ds = []
for i in range(len(X_train)):
    train_ds.append([np.transpose(X_train[i]), y_train[i]])

val_ds = []
for i in range(len(X_val)):
    val_ds.append([np.transpose(X_val[i]), y_val[i]])

bat_size = 64
print("\nNOTE:\nSetting batch-size to", bat_size)
train_ldr = torch.utils.data.DataLoader(train_ds, batch_size=bat_size, shuffle=True)
val_ldr = torch.utils.data.DataLoader(val_ds, batch_size=bat_size, shuffle=True)

energy = f.extract_energy_terms(X_train)

print(len(energy))
print(energy[0].shape)