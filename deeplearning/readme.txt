Before run this code, make sure there is a data folder, paralleling code folder, including test.csv, train.csv and embeddings folder.

How to run:
Open terminal, go to code directory, run the following command
Python run.py <network> <train> <submit>
For example:
Python run.py --network=CNN --train=True --submit=True 

network => which neural network to use: RNN (default) or CNN. Since the RNN network use CuDNNGRU that is GPU only layer, you have to run this code on computer with Nvidia Tesla GPUs
train => Whether to train a model or not. True: train model, False(default):load pre-tained model, if it is False, please make sure there is a model file(.h5) under the project directory
--submit => Where to run on test dataset and save the submission file for Kaggle. Default: False

