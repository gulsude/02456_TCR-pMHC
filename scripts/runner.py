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
dropOutRate = [0.1, 0.2, 0.3, 0.4, 0.5]
embeddings = ['Baseline', 'esm-1b']
current_time = time.strftime("%H:%M:%S", time.localtime())

for emb in embeddings:
    print('Running Notebook for {}'.format(emb))
    pm.execute_notebook(input_path='main2.ipynb',    
                        parameters={'embedding': emb},
                        output_path=NBDIR + '{}_{}_main2_encoding_{}.ipynb'.format(date, current_time, emb))