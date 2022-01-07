import pandas as pd
import numpy as np
import os
import random
import json, pickle
from collections import OrderedDict



#import networkx as nx
#pip install utils
#from utils import *



def extract_energy_terms(dataset_X):
    all_en = [np.concatenate((arr[0:190,20:], arr[192:,20:]), axis=0) for arr in dataset_X]  # 178
    return all_en

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
# nomarlize
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
data_list = []

import glob
for fp in glob.glob("../data/train/*input.npz"):
    data = np.load(fp)["arr_0"]
    data_list.append(data)
    targets = np.load(fp.replace("input", "labels"))["arr_0"]
    target_list.append(targets)
    
for fp in glob.glob("../data/validation/*input.npz"):
    data = np.load(fp)["arr_0"]
    data_list.append(data)
    targets = np.load(fp.replace("input", "labels"))["arr_0"]
    target_list.append(targets)
    
#print (len(target_list),len(target_list[0]),len(target_list[1]),len(target_list[2]),len(target_list[3]),len(target_list[4]))
#print(len(data_list))
#print(len(data_list[0]))

def energy_term(data_list):
    energy_sets=[]
    for i in range (len(data_list)):
        energy_set = extract_energy_terms(data_list[i]) 
        for j in range(0, len(energy_set)):
            pad = 420 - len(energy_set[j])
            energy_set[j] = np.pad(energy_set[j], ((0, pad), (0, 0)), 'constant')
        energy_sets.append(energy_set)
    return energy_sets



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
#print(seq_keys[0][0])
seq_lists=[]
for n in range(5):
    m = n+1
    seq_dir = os.path.join('GNN_data','data_all',str(m),'seq')
    seq_list=[]
    for i in range(len(seq_keys[n])):
        seq_file = os.path.join(seq_dir, str(seq_keys[n][i])+ '.fasta')
        infile = open(seq_file)
        for line in infile:
            if line.startswith('>'):
                pass
            else: 
                seq_list.append(line.strip())
    seq_lists.append(seq_list)
        

print("Sequences are extracted")

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
        #print(x)
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
#print(len(seq_keys))

seq_graphs=[]
for n in range(5):
    m = n+1
    aln_dir = os.path.join('GNN_data','data_all',str(m),'aln')
    pconsc4_dir = os.path.join('GNN_data','data_all',str(m), 'pconsc4')
    seq_graph = []
    for i in range(len(seq_keys[n])):
        
        g = target_to_graph(str(seq_keys[n][i]),seq_lists[n][i],pconsc4_dir,aln_dir)
        seq_graph.append(g)
    seq_graphs.append(seq_graph)

energy_set = energy_term(data_list)


en_train = np.concatenate(energy_set[1:4])
#print(len(en_train))
X_train = np.concatenate(seq_graphs[0:3])
#print(len(X_train))
y_train = np.concatenate(target_list[1:4])
#print(X_train[0],y_train[0])
#print(len(y_train))
en_valid = energy_set[0]
X_valid = seq_graphs[3]
y_valid = target_list[0]
#print(len(en_valid))
#print(len(X_valid))
en_test = energy_set[4]
X_test = seq_graphs[4]
y_test = target_list[4]

#print(len(X_train),len(y_train),len(X_valid),len(y_valid),len(X_test),len(y_test))
import torch
from torch_geometric.data import InMemoryDataset, DataLoader, Batch
from torch_geometric import data as DATA
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

torch.use_deterministic_algorithms(True)

def data_proccess(graph_data,y,en):
    data_list_pro = []
    #print(y[0],graph_data[0][0])
    for i in range(len(y)):
        #print(y[i])
        GCNData_pro = DATA.Data(x=torch.Tensor(graph_data[i][1]), en = torch.FloatTensor(en[i]),
                                    edge_index=torch.LongTensor(graph_data[i][2]).transpose(1, 0),
                                    y=torch.FloatTensor([y[i]]))
        GCNData_pro.__setitem__('target_size', torch.LongTensor([graph_data[i][0]]))
            
            
        data_list_pro.append(GCNData_pro)
    
    
    
    data_pro = data_list_pro
    
    loader = torch.utils.data.DataLoader(data_pro, batch_size= 512,shuffle=False,
                                              collate_fn=collate)
    return loader

    
def test_data_proccess(graph_data,y,en):
    data_list_pro = []
    #print(y[0],graph_data[0][0])
    for i in range(len(y)):
        #print(y[i])
        GCNData_pro = DATA.Data(x=torch.Tensor(graph_data[i][1]), en = torch.FloatTensor(en[i]),
                                    edge_index=torch.LongTensor(graph_data[i][2]).transpose(1, 0),
                                    y=torch.FloatTensor([y[i]]))
        GCNData_pro.__setitem__('target_size', torch.LongTensor([graph_data[i][0]]))
            
            
        data_list_pro.append(GCNData_pro)
    
    
    
    data_pro = data_list_pro
    
    loader = torch.utils.data.DataLoader(data_pro, batch_size= len(y),shuffle=False,
                                              collate_fn=collate)

    
    return loader


    test = data_proccess(X_test,y_test)

def get_mse(y, f):
    mse = ((y - f) ** 2).mean(axis=0)
    return mse



import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import roc_curve, confusion_matrix
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don’t have any parameters
from sklearn.metrics import accuracy_score, accuracy_score, roc_auc_score, roc_curve, auc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_max_pool as gmp, global_add_pool as gap,global_mean_pool as gep,global_sort_pool
from torch_geometric.utils import dropout_adj
from utils import *
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=300, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            #print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train_project(net, optimizer, train_ldr, val_ldr, test_ldr,X_valid, epochs, criterion, early_stop):
    num_epochs = epochs

    train_acc = []
    valid_acc = []
    test_acc = []

    train_losses = []
    valid_losses = []
    test_loss = []

    train_auc = []
    valid_auc = []
    test_auc = []

    no_epoch_improve = 0
    min_val_loss = np.Inf

    test_probs, test_preds, test_targs, test_peptides = [], [], [], []

    for epoch in range(num_epochs):
        cur_loss = 0
        val_loss = 0
        # Train
        net.train()
        train_preds, train_targs, train_probs = [], [], []
       
        for batch_idx, data in enumerate(train_ldr):
            data_pro = data.to(device)
            #print("data_pro", data_pro.size())
            
            optimizer.zero_grad()
            output = net(data_pro)
            batch_loss = criterion(output, data_pro.y.view(-1, 1).float().to(device))
            batch_loss.backward()
            optimizer.step()

            probs = torch.sigmoid(output.detach())
            preds = np.round(probs.cpu())
            train_probs += list(probs.data.cpu().numpy())
            train_targs += list(np.array(data_pro.y.view(-1, 1).float().to(device).cpu()))
            train_preds += list(preds.data.numpy())
            cur_loss += batch_loss.detach()

        train_losses.append(cur_loss / len(train_ldr.dataset))

        net.eval()
        # Validation
        val_preds, val_targs, val_probs = [], [], []
        with torch.no_grad():
            for batch_idx, data in enumerate(val_ldr):
                x_batch_val = data.to(device)

                output = net(x_batch_val)
                val_batch_loss = criterion(output, x_batch_val.y.view(-1, 1).float().to(device))

                probs = torch.sigmoid(output.detach())
                preds = np.round(probs.cpu())
                val_probs += list(probs.data.cpu().numpy())
                val_preds += list(preds.data.numpy())
                val_targs += list(np.array(x_batch_val.y.view(-1, 1).float().to(device).cpu()))
                val_loss += val_batch_loss.detach()

            valid_losses.append(val_loss / len(val_ldr.dataset))

            train_acc_cur = accuracy_score(train_targs, train_preds)
            valid_acc_cur = accuracy_score(val_targs, val_preds)
            train_auc_cur = roc_auc_score(train_targs, train_probs)
            valid_auc_cur = roc_auc_score(val_targs, val_probs)

            train_acc.append(train_acc_cur)
            valid_acc.append(valid_acc_cur)
            train_auc.append(train_auc_cur)
            valid_auc.append(valid_auc_cur)

        # Early stopping
        if (val_loss / len(X_valid)).item() < min_val_loss:
            no_epoch_improve = 0
            min_val_loss = (val_loss / len(X_valid))
        else:
            no_epoch_improve += 1
        if no_epoch_improve == early_stop:
            print("Early stopping\n")
            break

        if epoch % 5 == 0:
            print("Epoch {}".format(epoch),
                  " \t Train loss: {:.5f} \t Validation loss: {:.5f}".format(train_losses[-1], valid_losses[-1]))

    # Test
    if test_ldr != []:

        with torch.no_grad():
            for batch_idx, data in enumerate(test_ldr):
                #print(batch_idx)
                x_batch_test = data.to(device)


                output = net(x_batch_test)
                test_batch_loss = criterion(output, x_batch_test.y.view(-1, 1).float().to(device))

                probs = torch.sigmoid(output.detach())
                predsROC = probs
                preds = np.round(probs.cpu())
                test_probs = list(probs.data.cpu().numpy())
                test_preds = list(preds.data.numpy())
                test_predsROC = list(probs.data.cpu().numpy())
                #print("-----",test_predsROC)
                test_targs = list(np.array(x_batch_test.y.view(-1, 1).float().to(device).cpu()))
                test_loss = test_batch_loss.detach()
                #print(x_batch_test.y)
                test_auc_cur = roc_auc_score(test_targs, test_predsROC)
                test_acc_cur = accuracy_score(test_targs, test_preds)
                test_acc.append(test_acc_cur)
                #print(test_acc)
                test_auc.append(test_auc_cur)

    return train_acc, train_losses, train_auc, valid_acc, valid_losses, valid_auc, val_preds, val_targs, test_preds, list(
        test_targs), test_loss, test_acc, test_auc

# GCN based model
class GNNNet(torch.nn.Module):
    def __init__(self, n_output=1, num_features_pro=54, output_dim=128, dropout=0.5):
        super(GNNNet, self).__init__()



        # self.pro_conv1 = GCNConv(embed_dim, embed_dim)
        self.pro_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro_conv2 = GCNConv(num_features_pro, num_features_pro * 2)
        #self.pro_conv3 = GCNConv(num_features_pro * 2, num_features_pro * 4)
        # self.pro_conv4 = GCNConv(embed_dim * 4, embed_dim * 8)
        self.pro_fc_g1 = torch.nn.Linear(num_features_pro * 2, output_dim)
        #self.pro_fc_g2 = torch.nn.Linear(512, output_dim)

        self.bn0 = nn.BatchNorm1d(34)
        self.conv1 = nn.Conv1d(in_channels=34, out_channels=100, kernel_size=3, stride=2, padding=1)
        torch.nn.init.kaiming_uniform_(self.conv1.weight)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1_bn = nn.BatchNorm1d(100)

        self.conv2 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3, stride=2, padding=1)
        torch.nn.init.kaiming_uniform_(self.conv2.weight)
        self.conv2_bn = nn.BatchNorm1d(100)

        self.rnn = nn.LSTM(input_size=100,hidden_size=26,num_layers=3, dropout=0.1, batch_first=True, bidirectional = True)
        self.drop = nn.Dropout(p = 0.5)

        self.enfc1 = nn.Linear(100, 128)
        torch.nn.init.xavier_uniform_(self.enfc1.weight)

        self.softmax = nn.Softmax(dim=1)




        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # combined layers
        self.fc1 = nn.Linear(2*output_dim, 128)
        #self.fc2 = nn.Linear(512, 128)
        self.out = nn.Linear(128, 1)

    def forward(self,data_pro):
        
        # get protein input
        target_x, target_edge_index, target_batch,en = data_pro.x, data_pro.edge_index, data_pro.batch,data_pro.en.float().detach().requires_grad_(True).unsqueeze(2)
        #print(en.size())
        #print(target_x.size())
        # target_seq=data_pro.target

        # print('size')
        # print('mol_x', mol_x.size(), 'edge_index', mol_edge_index.size(), 'batch', mol_batch.size())
        # print('target_x', target_x.size(), 'target_edge_index', target_batch.size(), 'batch', target_batch.size())

       

        
        xt = self.pro_conv1(target_x, target_edge_index)
        xt = self.relu(xt)
        xt = self.dropout(xt)
        #print("xt", xt.size())
        # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
        xt = self.pro_conv2(xt, target_edge_index)
        xt = self.relu(xt)
        xt = self.dropout(xt)
        #print("xt", xt.size())
        # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
        #xt = self.pro_conv3(xt, target_edge_index)
        #xt = self.relu(xt)
        #xt = self.dropout(xt)
        #print("xt", xt.size())
        # xt = self.pro_conv4(xt, target_edge_index)
        # xt = self.relu(xt)
        xt = gep(xt, target_batch)  # global pooling
        #print("xt", xt.size())
        # flatten
        xt = self.relu(self.pro_fc_g1(xt))
        xt = self.dropout(xt)
        #print("xt", xt.size())
        #xt = self.pro_fc_g2(xt)
        #xt = self.relu(xt)
        #xt = self.dropout(xt)
        #print("xt", xt.size())

        x = self.bn0(en)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        #x = self.pool(x)
        #print("x", x.size())

        x = self.conv1_bn(x)
        #print("x", x.size())

        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.conv2_bn(x).squeeze(2)
        #print("x", x.size())
        #print(target_batch.size())
        #x = gep(x,torch.tensor(np.zeros((1755600, ),dtype=np.int)))
        batch = []
        #print(len(target_x))
        for i in range(int(len(en)/420)):
            for j in range(420):
                batch.append(i)

        batch_size= torch.tensor(batch)
        #print(batch_size)
        x = gep(x,batch_size)
        x = x.view(x.size(0), -1)
        
        x =self.enfc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        #print("x", x.size())
        #print("xt", xt.size())
        #print(target_batch.size())
        
        xc = torch.cat((x, xt),1)
        #print("xc", xc.size())
        #stop
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        print("xc", xc.size())
        #xc = xc.squeeze(2).transpose(2, 1)
        print("xc", xc.size())

        
        xc, (h, c) = self.rnn(xc)
        cat = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)
        cat = self.drop(cat)
        xc = self.fc1(cat)
        #xc = self.fc2(xc)
        #xc = self.relu(xc)
        #xc = self.dropout(xc)
        out = torch.sigmoid(self.out(xc))
        

    

        # print(x.size(), xt.size())
        # concat
        
        return out

TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.0001
NUM_EPOCHS = 2000


models_dir = 'models'
results_dir = 'results'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(results_dir):
    os.makedirs(results_dir)


result_str = ''
USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda:10.2' if USE_CUDA else 'cpu')
model = GNNNet()
model.to(device)

model_st = GNNNet.__name__




optimizer = torch.optim.Adam(model.parameters(), lr=LR)

#en_train_loader = torch.utils.data.DataLoader(en_train, batch_size= len(en_train),shuffle=False)
train_loader = data_proccess(X_train,y_train,en_train)
#print(train_loader)
#en_valid_loader = torch.utils.data.DataLoader(en_valid, batch_size= len(en_valid),shuffle=False)
valid_loader = data_proccess(X_valid,y_valid,en_valid)
#en_test_loader = torch.utils.data.DataLoader(en_test, batch_size= len(en_test),shuffle=False)
test_loader = test_data_proccess(X_test,y_test,en_test)


epochs = 1
patience=10
criterion = nn.BCEWithLogitsLoss()
train_acc, train_losses, train_auc, valid_acc, valid_losses, valid_auc, val_preds, val_targs, test_preds, test_targs, test_loss, test_acc, test_auc = train_project(model, optimizer, train_loader, valid_loader, test_loader,X_valid, epochs, criterion, patience)

print( test_acc, test_auc, train_acc, train_auc, valid_acc, valid_auc)

