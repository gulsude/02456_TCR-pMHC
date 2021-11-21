import os
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
dropOutRate = [0.1, 0.2, 0.3, 0.4, 0.5]
embeddings = ['Baseline', 'esm-1b']


for emb in embeddings:
    print('Running Notebook for {}'.format(emb))
    pm.execute_notebook(input_path='main2.ipynb',   #GIVES ERROR FOR KERNEL NAME
                        parameters={'embedding': emb},
                        output_path=NBDIR + '{}_main2_encoding_{}.ipynb'.format(date, emb))