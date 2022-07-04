# 02456_TCR-pMHC 2021
This is a repository for the final project of the course 02456 Deep Learning, Techical University of Denmark

## APPLYING LANGUAGE MODEL EMBEDDINGS TO IMPROVE PREDICTION OF TCR-PMHC INTERACTIONS

T-cell receptors (TCR) play an essential role in the immune response to identify and destroy pathogenic or pathogen- infected cells. The TCR recognize the antigens presented by the major histocompatibility complexes (MHC) on the cell membrane. Understanding and predicting the interactions be- tween TCR, antigens and MHC could boost the research for potential targets in cancer and vaccine research. We devel- oped a model that combines convolutional neural networks (CNN) and bidirectional long short-term memory (biLSTM) architecture, and the model is trained on several input data encoded with different strategies that uses embeddings from the pre-trained protein language models, and a grid search of optimal hyperparameters is applied. The model performances were measured by their accuracy, area under the curve (AUC), and Matthews correlation coefficient (MCC). We developed several more model architectures and compare their perfor- mance by training them on the data encoded with the best performing encoding method. Our approach leads better re- sults using ESM-1b as encoding for the separate sequences in comparison with the baseline model which uses sparse encoding. However the generalisation of the model still can be improved, and there is a clear necessity of enlarging the dataset.

_____________________________________________________________________________________

Data directories will be ignored while using git, therefore please place the data in:

data \
└───train\
└───validation
