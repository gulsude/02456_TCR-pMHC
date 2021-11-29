# different encodings here
# you need: pip install fair-esm

import pandas as pd
import numpy as np
import torch, sys
import esm
import gc
import DGraphDTAscripts as sp
import DGraphDTAdata_process as dp


def GNN_enc(peptide, pooling=False):
    


def esm_1b_peptide(peptide, pooling=False):
    #this only for one sentence
    peptides = [peptide]

    # Load pre-trained ESM-1b model

    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    data = []
    count = 0
    for peptide in peptides:
        data.append(("", peptide))
    batch_labels, batch_strs, batch_tokens = batch_converter(data)


    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33].numpy()

    sequence_representations = []
    del results, batch_labels, batch_strs, batch_tokens, model, alphabet, batch_converter
    gc.collect()

    for i, (_, seq) in enumerate(data):
        count += 1
        if count % 5 == 0:
            print("\t\tFlag", count)

        #pad = 420 - token_representations.shape[0]
        #token_representations = np.pad(token_representations, ((0, pad), (0, 0)), 'constant')
        if pooling:
            trp = token_representations[i, 1: len(seq) + 1].mean(0)
        else:
            trp = token_representations[i, 1: len(seq) + 1]
            pad = 420 - trp.shape[0]
            trp = np.pad(trp, ((0, pad), (0, 0)), 'constant')

    return trp

"""
def esm_1b_peptide(peptide, pooling=True):
    peptides = [peptide]
    embeddings = list()
    # Load pre-trained ESM-1b model

    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    data = []
    count = 0
    for peptide in peptides:
        data.append(("", peptide))
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33].numpy()[0]
    sequence_representations = []
    del results, batch_labels, batch_strs, batch_tokens, model, alphabet, batch_converter
    gc.collect()
    for i, (_, seq) in enumerate(data):
        count += 1
        if count % 100 == 0:
            print("\t\tFlag", count)
        pad = 420 - token_representations.shape[0]
        token_representations = np.pad(token_representations, ((0, pad), (0, 0)), 'constant')

        if pooling:
            return token_representations[i, 1: len(seq) + 1].mean(0)
        else:
            return token_representations[i, 1: len(seq) + 1]
"""

def esm_ASM(peptide, pooling=False):

    peptides = [peptide]

    # Load pre-trained ESM-MSA-1b model
    model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    batch_converter = alphabet.get_batch_converter()

    data = []
    for peptide in peptides:
        data.append(("", peptide))
    batch_labels, batch_strs, batch_tokens = batch_converter(data)


    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[12], return_contacts=True)
    token_representations = results["representations"][12].numpy()[0][0]

    del results, batch_labels, batch_strs, batch_tokens, model, alphabet, batch_converter
    gc.collect()
    sequence_representations = []

    for i, (_, seq) in enumerate(data):


        if pooling:
            sequence_representations = token_representations[1:, ].mean(0)

        else:
            sequence_representations = token_representations[1:, ]
            #add padding
            pad = 420 - sequence_representations.shape[0]
            sequence_representations = np.pad(sequence_representations, ((0, pad), (0, 0)), 'constant')

    return sequence_representations

# list of aa and list of properties in matrix aaIndex
aminoacidTp = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
aaProperties = ["hydrophobicity", "volume", "bulkiness", "polarity", "Isoelectric point", "coil freq", "bg freq"]

# loading matrices
bl50 = pd.read_csv("../data/Matrices/BLOSUM50", sep="\s+", comment="#", index_col=0)
bl50 = bl50.loc[aminoacidTp, aminoacidTp]
aaIndex = pd.read_csv("../data/Matrices/aaIndex.txt", sep=",", comment="#", index_col=0)
vhse = pd.read_csv("../data/Matrices/VHSE", sep="\s+", comment="#")

#standardizing blosum and aaIndex
mean = np.mean(bl50)
std = np.std(bl50)
bl50 = (bl50 - mean)/std

mean = np.mean(aaIndex, axis=1)
std = np.std(aaIndex, axis=1)
aaIndex = aaIndex.subtract(mean,axis='rows')
aaIndex = aaIndex.divide(std,axis='rows')

def encodePeptides(peptides, scheme, bias=False):
    # converting scheme to list if needed
    if type(scheme) != list:
        scheme = [scheme]
    if type(peptides) == str:
        peptides = [peptides]

    # encding by by aa/ by scheme

    enc_peptides = list()

    for peptide in peptides:
        seq = list()
        for aa in peptide:
            for sc in scheme:
                if sc == "blosum":
                    seq.append( bl50.loc[[aa]].values[0])
                elif sc == "allProperties":
                    seq.append(aaIndex[aa].values)
                elif sc == "vhse":
                    seq.append(vhse[aa].values)
                else:
                    print("ERROR: No encoding matrix with the name {}".format(sc))
        if bias:
            seq.append(1)
        seq = np.array(seq)
        print(seq)

        #padding to sequence:
        pad = 420 - seq.shape[0]
        seq = np.pad(seq, ((0,pad),(0,0)), 'constant')
        enc_peptides.append(seq)

    return enc_peptides


# the difference is the output shape - not used
def encodePeptidesCNN(peptides, scheme):
    # output
    encoded_pep = np.empty()

    # converting scheme to list if needed
    if type(scheme) != list:
        scheme = [scheme]

    # encding by peptide/by aa/ by scheme
    for peptide in peptides:
        pos = 0
        seq = []
        for aa in peptide:
            for sc in scheme:
                if sc == "blosum":
                    seq.append(bl50.loc[[aa]].values.tolist()[0])

                elif sc in aaProperties:
                    seq.append([aaIndex[aa][sc]])

                elif sc == "sparse":
                    seq.append(sp[aa].values.tolist())

                elif sc == "sparse2":
                    seq.append(sp2[aa].values.tolist())

                elif sc == "sparse3":
                    seq.append(sp3[aa].values.tolist())

                elif sc == "allProperties":
                    seq.append(aaIndex[aa].values.tolist())

                elif sc == "vhse":
                    seq.append(vhse[aa].values.tolist())

                elif sc == "pssm":
                    seq.append([pssm[aa][pos]])

                else:
                    print("ERROR: No encoding matrix with the name {}".format(sc))

            pos = pos + 1

    return encoded_pep
