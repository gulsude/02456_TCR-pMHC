import os
import time
import papermill as pm
from datetime import datetime

#initialize
date = datetime.today().strftime('%Y%m%d')
NBDIR = 'notebooks/'

try:
    os.mkdir(NBDIR)
except:
    print(NBDIR + 'directory already exists')

#hyperparameters
numHN = [32, 64, 128, 256]
numFilter = [50, 100, 200]
embeddings = ['Baseline', 'esm-1b', "blosum", 'vhse']
current_time = time.strftime("%H:%M:%S", time.localtime())

dropOutRate = 0.1  #Dayana
#dropOutRate = 0.2 #Gul
#dropOutRate = 0.3 #Enric
#dropOutRate = 0.4 #Shannara
#dropOutRate = 0.5 #Huijiao

for emb in embeddings:
    for nn in numHN:
        for nf in numFilter:

            print('Running Notebook for {}'.format(emb))
            pm.execute_notebook(input_path='main2.ipynb',
                                parameters={'embedding': emb, 'numHN':nn, 'numFilter':nf, 'dropOutRate': dropOutRate},
                                output_path=NBDIR + '{}_{}_main2_encoding_{}_numHN_{}_filters_{}_dr_{}.ipynb'.format(date, current_time, emb, nn, nf, dropOutRate))