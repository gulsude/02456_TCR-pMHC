import pandas as pd
import numpy as np
import os
import random
import json, pickle
from collections import OrderedDict

import networkx as nx

from utils import *

def return_aa(one_hot):
    mapping = dict(zip(range(20),"ACDEFGHIKLMNPQRSTVWY"))
    try:
        index = one_hot.index(1)
        return mapping[index]     
    except:
        return 'X'

def reverseOneHot(encoding):
    """
    Converts one-hot encoded array back to string sequence
    """
    seq=''
    for i in range(len(encoding)):
            if return_aa(encoding[i].tolist()) != 'X':
                seq+=return_aa(encoding[i].tolist())
    return seq

def extract_sequences(dataset_X, merge=False):
    """
    Return DataFrame with MHC, peptide and TCR a/b sequences from
    one-hot encoded complex sequences in dataset X
    """
    mhc_sequences = [reverseOneHot(arr[0:179,0:20]) for arr in dataset_X]
    pep_sequences = [reverseOneHot(arr[179:192,0:20]) for arr in dataset_X] ## 190 or 192 ????
    tcr_sequences = [reverseOneHot(arr[192:,0:20]) for arr in dataset_X]
    all_sequences = [reverseOneHot(arr[179:192,0:20]) for arr in dataset_X]

    if merge:
        df_sequences = pd.DataFrame({"all": all_sequences})

    else:
        df_sequences = pd.DataFrame({"MHC":mhc_sequences,
                                 "peptide":pep_sequences,
                                 "TCR":tcr_sequences})
        
    return df_sequences    


def dic_normalize(dic):
    # print(dic)
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    # print(max_value)
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic



target_list = []

import glob
for fp in glob.glob("../data/train/*input.npz"):
   
    targets = np.load(fp.replace("input", "labels"))["arr_0"]
    
    target_list.append(targets)
    
for fp in glob.glob("../data/validation/*input.npz"):
    
    targets = np.load(fp.replace("input", "labels"))["arr_0"]
    
    target_list.append(targets)
    


def produced_key(n):
    seq_key=[]
    for i in range(n):
        seq_key.append(i)
    return seq_key

seq_keys=[]
seq_keys.append(produced_key(1480))
seq_keys.append(produced_key(1532))
seq_keys.append(produced_key(1168))
seq_keys.append(produced_key(1526))
seq_keys.append(produced_key(1207))
print(seq_keys[0][0])
seq_lists=[]
for n in range(5):
    m = n+1
    seq_dir = os.path.join('GNN_data','data',str(m),'seq')
    seq_list=[]
    for i in range(len(seq_keys[n])):
        seq_file = os.path.join(seq_dir, str(seq_keys[n][i])+ '.fasta')
        infile = open(seq_file)
        for line in infile:
            if line.startswith('>'):
                pass
            else: 
                seq_list.append(line.strip())
        #seq_list.append(infile.read()[3:-1])
    seq_lists.append(seq_list)
        
seq_lists[0][1220]

pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']

pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}

res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}

res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}
res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}

res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)

# one ont encoding
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def residue_features(residue):
    res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                     1 if residue in pro_res_polar_neutral_table else 0,
                     1 if residue in pro_res_acidic_charged_table else 0,
                     1 if residue in pro_res_basic_charged_table else 0]
    res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue], res_pkx_table[residue],
                     res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
    # print(np.array(res_property1 + res_property2).shape)
    return np.array(res_property1 + res_property2)


# target feature for target graph
def PSSM_calculation(aln_file, pro_seq):
    pfm_mat = np.zeros((len(pro_res_table), len(pro_seq)))
    with open(aln_file, 'r') as f:
        line_count = len(f.readlines())
        for line in f.readlines():
            if len(line) != len(pro_seq):
                print('error', len(line), len(pro_seq))
                continue
            count = 0
            for res in line:
                if res not in pro_res_table:
                    count += 1
                    continue
                pfm_mat[pro_res_table.index(res), count] += 1
                count += 1
    # ppm_mat = pfm_mat / float(line_count)
    pseudocount = 0.8
    ppm_mat = (pfm_mat + pseudocount / 4) / (float(line_count) + pseudocount)
    pssm_mat = ppm_mat
    # k = float(len(pro_res_table))
    # pwm_mat = np.log2(ppm_mat / (1.0 / k))
    # pssm_mat = pwm_mat
    # print(pssm_mat)
    return pssm_mat

def seq_feature(pro_seq):
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
    pro_property = np.zeros((len(pro_seq), 12))
    for i in range(len(pro_seq)):
        # if 'X' in pro_seq:
        #     print(pro_seq)
        pro_hot[i,] = one_of_k_encoding(pro_seq[i], pro_res_table)
        pro_property[i,] = residue_features(pro_seq[i])
    return np.concatenate((pro_hot, pro_property), axis=1)

def target_feature(aln_file, pro_seq):
    pssm = PSSM_calculation(aln_file, pro_seq)
    other_feature = seq_feature(pro_seq)
    # print('target_feature')
    # print(pssm.shape)
    # print(other_feature.shape)

    # print(other_feature.shape)
    # return other_feature
    return np.concatenate((np.transpose(pssm, (1, 0)), other_feature), axis=1)

# target aln file save in data/dataset/aln
def target_to_feature(target_key, target_sequence, aln_dir):
    # aln_dir = 'data/' + dataset + '/aln'
    aln_file = os.path.join(aln_dir, target_key + '.aln')
    # if 'X' in target_sequence:
    #     print(target_key)
    feature = target_feature(aln_file, target_sequence)
    return feature

# pconsc4 predicted contact map save in data/dataset/pconsc4
def target_to_graph(target_key, target_sequence, contact_dir, aln_dir):
    target_edge_index = []
    target_size = len(target_sequence)
    # contact_dir = 'data/' + dataset + '/pconsc4'
    contact_file = os.path.join(contact_dir, target_key + '.npy')
    contact_map = np.load(contact_file)
    contact_map += np.matrix(np.eye(contact_map.shape[0]))
    index_row, index_col = np.where(contact_map >= 0.5)
    for i, j in zip(index_row, index_col):
        target_edge_index.append([i, j])
    target_feature = target_to_feature(target_key, target_sequence, aln_dir)
    target_edge_index = np.array(target_edge_index)
    return target_size, target_feature, target_edge_index

seq_graphs=[]
for n in range(5):
    m = n+1
    aln_dir = os.path.join('GNN_data','data',str(m),'aln')
    pconsc4_dir = os.path.join('GNN_data','data',str(m), 'pconsc4')
    seq_graph = []
    for i in range(len(seq_keys[n])):
        g = target_to_graph(str(seq_keys[n][i]),seq_lists[n][i],pconsc4_dir,aln_dir)
        
        seq_graph.append(g)
    seq_graphs.append(seq_graph)

X_train = np.concatenate(seq_graphs[0:3])
y_train = np.concatenate(target_list[0:3])
print(X_train[0][0])
X_valid = np.concatenate(seq_graphs[3:4])
y_valid = np.concatenate(target_list[3:4])

X_test = np.concatenate(seq_graphs[4:])
y_test = np.concatenate(target_list[4:])

from torch_geometric.data import InMemoryDataset, DataLoader, Batch
from torch_geometric import data as DATA
def data_proccess(graph_data,y):
    data_list_pro = []
    for i in range(len(graph_data)):
        GCNData_pro = DATA.Data(x=torch.Tensor(graph_data[i][1]),
                                    edge_index=torch.LongTensor(graph_data[i][2]).transpose(1, 0),
                                    y=torch.FloatTensor([y[i]]))
        GCNData_pro.__setitem__('target_size', torch.LongTensor([graph_data[i][0]]))
            
            
        data_list_pro.append(GCNData_pro)
    
    
    
    data_pro = data_list_pro
    
    loader = torch.utils.data.DataLoader(data_pro, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                              collate_fn=collate)

    
    return loader

