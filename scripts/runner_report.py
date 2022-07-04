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
esm_1b_separated = [True, False]  ### upd
embedding = [ 'esm-1b','esm_ASM']  
numHN = [26, 32, 64]
numFilter = [50, 100, 200]
learning_rate=[0.001, 0.0005]
weight_decay = [0.0001, 0.0005]
dropOutRate = [0.1, 0.2]

#for ML- flow
name_experiment = "hyperparameter grid"


for sep in esm_1b_separated:
    for emb in embedding:
        for nn in numHN:
            for nf in numFilter:
                for lr in learning_rate:
                    for wd in weight_decay:
                        for do in dropOutRate:
                            if emb == 'esm_ASM' and sep == True:
                                continue
                            else:
                                notebook_name = '{}_{}_main3_encoding_{}_numHN_{}_filters_{}_dr_{}_lr_{}_wc_{}_separated_{}.ipynb'.format(date, current_time, emb, nn, nf, (int(do*10)), str(lr).replace(".",""), str(wd).replace(".",""), sep)
                                print("Running:", notebook_name)

                                pm.execute_notebook(input_path='main3.ipynb',
                                                    parameters={'embedding': emb,
                                                                'numHN':nn,
                                                                'numFilter': nf,
                                                                'dropOutRate': do,
                                                                'esm_1b_separated': sep,  ### upd
                                                                'learning_rate': lr,
                                                                'weight_decay': wd,
                                                                'name_experiment': name_experiment
                                                                },
                                            output_path=NBDIR + notebook_name
                                            )
