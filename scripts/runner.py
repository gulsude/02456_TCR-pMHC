import os
import time
import papermill as pm
from datetime import datetime

#initialize
date = datetime.today().strftime('%Y%m%d')
current_time = time.strftime("%H-%M-%S", time.localtime())
NBDIR = 'notebooks/'

try:
    os.mkdir(NBDIR)
except:
    print(NBDIR + 'directory already exists')

#hyperparameters
numHN = [32, 64, 128, 256] #
numFilter = [50, 100, 200]
embeddings = ["blosum", 'Baseline', 'esm-1b','vhse' ,"esm_ASM"]
keep_energy=True

#dropOutRate = 0.1  #Dayana
#dropOutRate = 0.2 #Gul
#dropOutRate = 0.3 #Enric
#dropOutRate = 0.4 #Shannara
#dropOutRate = 0.5 #Huijiao

for emb in embeddings:
    for nn in numHN:
        for nf in numFilter:

            print('Running Notebook for {}'.format(emb))
            pm.execute_notebook(input_path='main2.ipynb',
                                parameters={'embedding': emb,
                                            'numHN':nn,
                                            'numFilter':nf,
                                            'dropOutRate': dropOutRate,
                                            'keep_energy': keep_energy},
                                output_path=NBDIR + '{}_{}_main2_encoding_{}_numHN_{}_filters_{}_dr_{}_keep_energy_{}.ipynb'.format(date, current_time, emb, nn, nf, (int(dropOutRate*10)),keep_energy))
