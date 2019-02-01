from echos import *
import numpy as np
import medpy.filter.smoothing as mp
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.animation import FuncAnimation
import torch
import matplotlib.pyplot as plt
import cv2
from parser import ConfigParserEcho
from source.segment.rnmf_segment import SegRNMF
from source.utils import *
import argparse
import time
from sklearn.decomposition import NMF


#########################################################################################################

# test seg method
# dir = '/Users/jesseprovost/Downloads/echo/out/mitral_valve/nnmf/4CH/'
# dir = '/Users/jesseprovost/Downloads/echo/out/mitral_valve/nnmf/4CH (-d 0 -dp 10)/'
# dir = '/Users/jesseprovost/Downloads/echo/out/mitral_valve/nnmf/4CH (-d 5 -dp 10 -b 0.1 -sc 0.1) 25 hidden/'
# dir = '/Users/jesseprovost/Downloads/echo/out/mitral_valve/rnmf/optical_flow/4CH/'
dir = '/Users/jesseprovost/Downloads/echo/out/mitral_valve/rnmf/original/4CH/'

# get list with all frames
func_walk = os.walk('/Users/jesseprovost/Downloads/echo/out/mitral_valve/frames/functional')
deg_walk = os.walk('/Users/jesseprovost/Downloads/echo/out/mitral_valve/frames/degenerative')

func_list = []
for path, _, _ in func_walk:
    func_list.append(path)

deg_list = []
for path, _, _ in deg_walk:
    deg_list.append(path)

deg_list = deg_list[1:]
func_list = func_list[1:]

deg_list.extend(func_list)
patient_list = deg_list

mask_dir = '/Users/jesseprovost/Downloads/echo/out/mitral_valve/frame_test'
box_dir = '/Users/jesseprovost/Downloads/echo/out/mitral_valve/window_new'

window_list = []
# for files, _, _ in os.walk(box_dir):
#     window_list.append(files)

# window_list = window_list[1:]

precision = []
recall = []
dice = []
window_percent = []
int_union = []

for i in range(len(patient_list)):
    window_list.append(box_dir + '/' + patient_list[i].split('/')[-1])

patients = []
print(patient_list)
print(window_list)

for i in range(len(patient_list)):
    # patient_list[i] = '/Users/jesseprovost/Downloads/echo/out/mitral_valve/frames/degenerative/88'
    # patient_list[i] = '/Users/jesseprovost/Downloads/echo/out/mitral_valve/frames/functional/17'
    patient = patient_list[i].split('/')[-1]
    print(patient)
    patients.append(patient)
    frame_list = os.listdir(patient_list[i])
    mask_list = []
    mask_ind = []
    # return frames with masks
    for j in range(len(frame_list)):
        if 'mask' in frame_list[j]:
            mask_list.append(frame_list[j])
            mask_ind.append(int(frame_list[j].split('_')[0]))
        else:
            pass

    #################################################################
    # load sparse matrix/valve

    if 'rnmf' in dir:
        x_hat = np.load(dir + patient + '/x_hat.npy')
        s_new = np.load(dir + patient + '/s.npy')
        mask = np.load(dir + patient + '/mask.npy')
    else:
        s = torch.load(dir + patient + '/s.pt').detach().numpy()
        x_hat = torch.load(dir + patient + '/x_hat.pt').detach().numpy()
        x_hat = np.reshape(x_hat, newshape=(400, 400, s.shape[1]))

        s = thresholding_fn(s, 99)
        s = np.reshape(s, newshape=(400, 400, s.shape[1]))

        s_aniso = np.empty_like(s)
        for j in range(s.shape[2]):
            s_aniso[:, :, j] = mp.anisotropic_diffusion(s[:, :, j], niter=5, kappa=20)

        s_aniso[s_aniso > 0] = 1
        flow = optical_flow(s, winsize=20)
        mask = window_detection(flow, False, window_size=(60, 80), num_frames=s.shape[2])
        s_new = mask * s_aniso

    num = []
    lst1_len = []
    lst2_len = []

    # load masks
    for k in range(3):
        img = np.load(mask_dir + '/' + patient + '/' + str(mask_ind[k]) + '.npy')

        # plt.subplot(1, 2, 1)
        # plt.imshow(img)
        # plt.subplot(1, 2, 2)
        # plt.imshow(s_new[:, :, mask_ind[k]-1])
        # plt.show()

        # calculate intersection
        lst1 = np.where(img.flatten() != 0)
        lst2 = np.where(s_new[:, :, mask_ind[k]-1].flatten() != 0)
        num.append(len(np.intersect1d(lst1, lst2)))
        lst1_len.append(len(lst1[0]))
        lst2_len.append(len(lst2[0]))

    precision.append((sum(num)) / sum(lst2_len))
    recall.append(sum(num) / sum(lst1_len))
    dice.append((2 * sum(num)) / (sum(lst1_len) + sum(lst2_len)))
    print(patient)
    print(precision[i])
    print(recall[i])
    print(dice[i])

    # window detection accuracy
    window = np.load(window_list[i] + '/box.npy')

    # plt.subplot(1, 2, 1)
    # plt.imshow(mask[:, :, 0])
    # plt.subplot(1, 2, 2)
    # plt.imshow(window)
    # plt.show()

    mask_size = len(np.where(mask[:, :, 0].flatten() != 0)[0])
    lst1 = np.where(window.flatten() != 0)
    lst2 = np.where(mask[:, :, 0].flatten() != 0)
    inter = len(np.intersect1d(lst1, lst2))
    union = len(np.union1d(lst1, lst2))
    window_percent.append(inter / mask_size)
    print(window_percent[i])
    int_union.append(inter/union)

dir = dir + 'results_new/'

if not os.path.exists(dir):
    os.makedirs(dir)

np.save(file=dir + 'precision.npy', arr=precision)
np.save(file=dir + 'recall.npy', arr=recall)
np.save(file=dir + 'dice.npy', arr=dice)
np.save(file=dir + 'window_percent.npy', arr=window_percent)
np.save(file=dir + 'int_union.npy', arr=int_union)

print('window_percent:', window_percent)
print(precision)
print('precision mean:', np.mean(precision))
print(recall)
print('recall mean:', np.mean(recall))
print(dice)
print('dice mean:', np.mean(dice))
print(patients)
print('intersection over union:', np.mean(int_union))







