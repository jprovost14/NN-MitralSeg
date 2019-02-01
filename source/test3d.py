from echos import *
import os
import time
import sys
sys.path.append(r'/local/home/jprovost/echo')
import numpy as np
from parser import ConfigParserEcho
import argparse
from source.segment.rnmf_segment import SegRNMF
from source.segment.nnmf_segment import SegNNMF
import torch


def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


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
    parser.add_argument('-y', '--config', default='/local/home/jprovost/echo/configurations/test3d.ini')
    parser.add_argument('-lr', '--learningrate', default=1e-4)
    parser.add_argument('-d', '--dim', default=0)
    parser.add_argument('-dp', '--dimp', default=2)
    parser.add_argument('-sc', '--sparsecoef', default=0.5)
    parser.add_argument('-b', '--beta', default=0.5)
    parser.add_argument('-ep', '--epochs', default=5)
    parser.add_argument('-nmf', '--nmf_init', default=0)
    args = parser.parse_args()

    epochs = args.epochs
    learning_rate = args.learningrate
    d = args.dim
    dprime = args.dimp
    sparsity_coef = args.sparsecoef
    beta = args.beta
    nmf_init = args.nmf_init

    conf = ConfigParserEcho()
    print('conf.read:', args.config)
    config_file = conf.read(args.config)[0]
    conf.copy_conf(config_file, time)

    # patient_list = os.listdir('/Users/jesseprovost/Downloads/echo/out/mitral_valve/pickle/4CH')
    patient_list = os.listdir('/local/home/jprovost/echo/out/mitral_valve/pickle/4CH')
    # patient_list = patient_list[37:]
    # patient_list = ['DEGENERATIVEDEGENERATIVE72 BIRRER-GRAF 4CH.pkl']

    print(patient_list)

    for i in range(len(patient_list)):
        patient_id = '4CH/' + patient_list[i][:-4]
        print(patient_id)
        # patient_id = '4CH/DEGENERATIVEDEGENERATIVE72 BIRRER-GRAF 4CH'
        # patient_id = '4CH/17'

        dt = DataCollection(*conf.get_par_load_save())
        if conf.getboolean('Load_Save', 'load_dataset'):
            dt = dt.load_pickle(patient_id)
            print(patient_id)
        if not dt.populated:
            dt.populate()
            processor = EchoProcess(*conf.get_par_video_processing())
            processor.process_dataset(dt)

        x = dt.matrix3d
        x = np.nan_to_num(x)

        seg = SegNNMF(matrix3d=x, sparsity_coef=float(sparsity_coef), beta=float(beta), epochs=int(epochs),
                      learning_rate=float(learning_rate), d=int(d), dprime=int(dprime), batchsize=200000,
                      num_workers=4, save_loc=patient_id, device=device, nmf_init=bool(int(nmf_init)))
        seg.train()

        # seg = SegRNMF(matrix3d=x, rank=2, sparsity_coef=(0.2, 0.001), save_loc=patient_id,
        #               option='rnmf_seg', max_iter=20, thresh1=99, thresh2=99.2)
        # seg.segment()

