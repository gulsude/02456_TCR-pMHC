# different encodings here
# you need: pip install fair-esm

import pandas as pd
import numpy as np
import torch
import esm

# Load pre-trained ESM-1b model
model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
batch_converter = alphabet.get_batch_converter()

# Load pre-trained ESM-MSA-1b model
modelMSA, alphabet2 = esm.pretrained.esm_msa1b_t12_100M_UR50S()
batch_converter = alphabet.get_batch_converter()

# list of aa and list of properties in matrix aaIndex
aminoacidTp = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
aaProperties = ["hydrophobicity", "volume", "bulkiness", "polarity", "Isoelectric point", "coil freq", "bg freq"]


def encodePeptides(peptides, scheme, bias=False):
    # loading matrices
    bl50 = pd.read_csv("../data/Matrices/BLOSUM50", sep="\s+", comment="#", index_col=0)
    bl50 = bl50.loc[aminoacidTp, aminoacidTp]
    aaIndex = pd.read_csv("../data/Matrices/aaIndex.txt", sep=",", comment="#", index_col=0)
    sp = pd.read_csv("../data/Matrices/sparse", sep=" ", comment="#", header=None)
    sp.columns = aminoacidTp
    sp2 = pd.read_csv("../data/Matrices/sparse2", sep=" ", comment="#", header=None).astype(float)
    sp2.columns = aminoacidTp
    sp3 = pd.read_csv("../data/Matrices/sparse3", sep=" ", comment="#", header=None).astype(float)
    sp3.columns = aminoacidTp
    vhse = pd.read_csv("../data/Matrices/VHSE", sep="\s+", comment="#")
    pssm = pd.read_csv("../data/Matrices/pssm", sep="\t", comment="#")

    # output
    encoded_pep = []

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
                    seq += bl50.loc[[aa]].values.tolist()[0]

                elif sc in aaProperties:
                    seq.append(aaIndex[aa][sc])

                elif sc == "sparse":
                    seq += sp[aa].values.tolist()

                elif sc == "sparse2":
                    seq += sp2[aa].values.tolist()

                elif sc == "sparse3":
                    seq += sp3[aa].values.tolist()

                elif sc == "allProperties":
                    seq += aaIndex[aa].values.tolist()

                elif sc == "vhse":
                    seq += vhse[aa].values.tolist()

                elif sc == "pssm":
                    seq.append(pssm[aa][pos])

                else:
                    print("ERROR: No encoding matrix with the name {}".format(sc))

            pos = pos + 1
        if bias:
            seq.append(1)
        encoded_pep.append(seq)
    return encoded_pep


# the difference is the output shape
def encodePeptidesCNN(peptides, scheme):
    # loading matrices
    bl50 = pd.read_csv("../data/Matrices/BLOSUM50", sep="\s+", comment="#", index_col=0)
    bl50 = bl50.loc[aminoacidTp, aminoacidTp]
    aaIndex = pd.read_csv("../data/Matrices/aaIndex.txt", sep=",", comment="#", index_col=0)
    sp = pd.read_csv("../data/Matrices/sparse", sep=" ", comment="#", header=None)
    sp.columns = aminoacidTp
    sp2 = pd.read_csv("../data/Matrices/sparse2", sep=" ", comment="#", header=None).astype(float)
    sp2.columns = aminoacidTp
    sp3 = pd.read_csv("../data/Matrices/sparse3", sep=" ", comment="#", header=None).astype(float)
    sp3.columns = aminoacidTp
    vhse = pd.read_csv("../data/Matrices/VHSE", sep="\s+", comment="#")
    pssm = pd.read_csv("../data/Matrices/pssm", sep="\t", comment="#")

    # output
    encoded_pep = []

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
        encoded_pep.append(seq)
    return encoded_pep