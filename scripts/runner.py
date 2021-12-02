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
embedding = [ 'esm-1b',"esm_ASM" ]
keep_energy=True
numHN = [26, 64]
numFilter = [100, 200]
learning_rate=[0.001]
weight_decay = [0.0001, 0.0005]
dropOutRate = [0.1, 0.3]


#for ML- flow
name_experiment = "hyperparameter grid"

for emb in embedding:
    for nn in numHN:
        for nf in numFilter:
            for lr in learning_rate:
                for wd in weight_decay:
                    for do in dropOutRate:
                        print('Running Notebook for {}'.format(emb))
                        pm.execute_notebook(input_path='main2.ipynb',
                                            parameters={'embedding': emb,
                                                        'numHN':nn,
                                                        'numFilter': nf,
                                                        'dropOutRate': do,
                                                        'keep_energy': keep_energy,
                                                        'learning_rate': lr,
                                                        'weight_decay': wd,
                                                        'name_experiment': name_experiment
                                                        },
                                            output_path=NBDIR + '{}_{}_main2_encoding_{}_numHN_{}_filters_{}_dr_{}_keep_energy_{}_lr_{}_wc_{}.ipynb'.format(
                                                date, current_time, emb, nn, nf, (int(do*10)),keep_energy, str(lr).replace(".",""), str(wd).replace(".",""))
                                            )
