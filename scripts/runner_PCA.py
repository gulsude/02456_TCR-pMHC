import os
import time
import papermill as pm
from datetime import datetime

#initialize
date = datetime.today().strftime('%Y%m%d')
current_time = time.strftime("%H-%M-%S", time.localtime())
NBDIR = 'notebooks/pca_models'



try:
    os.mkdir('notebooks')
    os.mkdir(NBDIR)
except:
    print(NBDIR + 'directory already exists')

#hyperparameters
modelName = "Net_project"
embedding = 'esm-1b'
numHN = [26,  64]
numFilter = [100, 200]
learning_rate= 0.001
weight_decay = 0.0001
dropOutRate = [0.5, 0.3]

#for ML- flow
name_experiment = "PCA"

for filter_num in numFilter:
    for nn in numHN:
        for do in dropOutRate:
            notebook_name = '{}_{}_model_test_{}_numHN_{}_filters_{}_dr_{}_lr_{}_wc_{}.ipynb'.format(date,current_time, modelName, nn, filter_num, str(do).replace(".",""),str(learning_rate).replace(".",""),str(weight_decay).replace(".",""))
            print("Running:", notebook_name)

            pm.execute_notebook(input_path='main_PCA.ipynb',
                                parameters={'embedding': embedding,
                                            'numHN': nn,
                                            'numFilter': filter_num,
                                            'dropOutRate': do,
                                            'learning_rate': learning_rate,
                                            'weight_decay': weight_decay,
                                            'name_experiment': name_experiment,
                                            'modelName' : modelName
                                            },
                                output_path=NBDIR + notebook_name
                                )

        print(datetime.now())

