from echos import *
import os
import time
import sys
sys.path.append(r'/local/home/jprovost/echo')
import numpy as np
import configparser
import argparse
from source.segment.rnmf_segment import SegRNMF
from source.segment.nnmf_segment import SegNNMF
import torch
from source.utils import *
import pickle


if __name__ == '__main__':

    # current time for file names
    time = time.strftime("%Y%m%d-%H%M%S")
    print("Time:", time)

    # check if gpu is available
    if torch.cuda.is_available():
        device = 'cuda:' + str(get_free_gpu())
    else:
        device = 'cpu'
    print('INFO: Start on device %s' % device)

    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--config', default='/Users/jesseprovost/Downloads/'
                                                  'NN-MitralSeg/configurations/test_config.ini')
    args = parser.parse_args()

    conf = args.config
    config = configparser.ConfigParser()
    config.read(conf)

    # factorization type
    fact_type = config['Parameters']['fact_type']

    # number of epochs
    epochs = config['Parameters']['epochs']

    # learning rate
    learning_rate = config['Parameters']['learning_rate']

    # d dimension in paper
    d = config['Parameters']['dim']

    # d prime dimension in paper
    dprime = config['Parameters']['dim_prime']

    # sparsity coefficient
    sparsity_coef = config['Parameters']['sparsity_coef']

    # l2 coefficient
    beta = config['Parameters']['beta']

    # boolean to initialize latent dims with nonnegative matrix factorization
    nmf_init = config['Parameters']['nmf_init']

    # data folder
    data_folder = config['Parameters']['data_folder']

    # number of layers and nodes in sparse network
    s_layers = config['Parameters']['s_layers']
    s_nodes = config['Parameters']['s_nodes']

    # number of layers and nodes in reconstruction network
    x_layers = config['Parameters']['x_layers']
    x_nodes = config['Parameters']['x_nodes']

    # get list of the echos in the data folder
    patient_list = os.listdir(str(data_folder))
    print(patient_list)

    # loop over all patients and run nnmf or rnmf
    for i in range(len(patient_list)):
        patient_id = str(data_folder) + patient_list[i]

        with open(patient_id, 'rb') as f:
            dt = pickle.load(f)

        # TODO check type of loaded pickle and ensure its a three dimensional numpy array

        x = dt.matrix3d

        # fill NaNs with 0s
        x = np.nan_to_num(x)

        if str(fact_type) == 'nnmf':

            # run NN-MitralSeg algorithm
            seg = SegNNMF(matrix3d=x, sparsity_coef=float(sparsity_coef), beta=float(beta), epochs=int(epochs),
                          learning_rate=float(learning_rate), d=int(d), dprime=int(dprime), batchsize=200000,
                          num_workers=4, save_loc=patient_id, device=device, nmf_init=bool(int(nmf_init)),
                          s_layers=int(s_layers), s_nodes=int(s_nodes), x_layers=int(x_layers), x_nodes=int(x_nodes))
            seg.train()
        else:

            # run robust nonnegative matrix factorization as implemented in [citation]
            seg = SegRNMF(matrix3d=x, rank=2, sparsity_coef=(0.2, 0.001), save_loc=patient_id,
                          option='rnmf_seg', max_iter=20, thresh1=99, thresh2=99.2)
            seg.segment()
