data_list = []
target_list = []
import torch
import numpy as np

import encoding as enc
import functions as func
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

X_val = np.concatenate(data_list[-1:])
y_val = np.concatenate(target_list[-1:])
X_val = X_val[:5]
nsamples, nx, ny = X_val.shape
print("Val set shape:", nsamples, nx, ny)


embedding="esm-1b"
data_list=[X_val]

print("-----------")

data_list_enc = []
if embedding == "Baseline":
    data_list_enc = data_list

elif embedding == "esm-1b":
    for dataset in data_list:
        print(type(dataset))
        print(dataset.shape)
        x_enc = np.array(func.extract_sequences(dataset, merge=True).values.tolist())
        print("x_enc1")
        print(x_enc.shape)
        x_enc = np.array([enc.esm_1b(seq, pooling=False) for seq in x_enc])
        print("x_enc2")
        print(x_enc)
        print(x_enc.shape)
        data_list_enc.append(x_enc)


elif embedding == "esm_ASM":
    for dataset in data_list:
        x_enc = func.extract_sequences(dataset, merge=True).values.tolist()
        x_enc = [enc.esm_ASM(seq, pooling=False) for seq in x_enc]
        data_list_enc.append(x_enc)

else:
    print("flag")
    for dataset in data_list:
        x_enc = func.extract_sequences(dataset, merge=True).values.tolist()
        #print(x_enc)
        x_enc = [enc.encodePeptidesCNN(seq, scheme=embedding) for seq in x_enc]
        print(x_enc)
        data_list_enc.append(x_enc)


print(len(data_list_enc))
print(len(data_list_enc[0]))
print(len(data_list_enc[0][0]))
print(len(data_list_enc[0][0][0]))




#energy = f.extract_energy_terms(X_val)

