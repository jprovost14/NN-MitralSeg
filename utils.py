import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
import numpy as np


def animate(tensor):
    fig, ax = plt.subplots(figsize=(8, 8))

    def update(j):
        ax.imshow(tensor[:, :, j])
        ax.set_axis_off()

    anim = FuncAnimation(fig, update, frames=tensor.shape[-1], interval=1)
    plt.show()
    return


def otsu(matrix):
    for i in range(matrix.shape[2]):
        matrix[:, :, i] = cv2.GaussianBlur(np.array(matrix[:, :, i] * 255, dtype=np.uint8), (15, 15), sigmaX=0.5,
                                           sigmaY=0.5)
        _, matrix[:, :, i] = cv2.threshold(np.array(matrix[:, :, i], dtype=np.uint8), 0, 255,
                                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return matrix


def optical_flow(tensor, winsize):
    flow = np.zeros(shape=(tensor.shape[0], tensor.shape[1], tensor.shape[2]-1))
    for i in range(tensor.shape[2]-1):
        flow_val = cv2.calcOpticalFlowFarneback(tensor[:, :, i], tensor[:, :, i+1], flow=None, pyr_scale=0.5, levels=1,
                                                winsize=winsize, iterations=3, poly_n=7, poly_sigma=3.5, flags=1)
        flow[:, :, i] = np.sqrt(np.square(flow_val[:, :, 1]) + np.square(flow_val[:, :, 0]))

    return flow


def thresholding_fn(matrix, thresh):
    # calculate threshold value for given percentile
    thresh_val = np.percentile(matrix, q=thresh)
    # assign binary values to pixels above and below threshold
    matrix[matrix < thresh_val] = matrix[matrix < thresh_val] * 0
    return matrix


def denoise(h, matrix):
    denoised = np.empty_like(matrix)

    for i in range(matrix.shape[2]):
        denoised[:, :, i] = cv2.fastNlMeansDenoising(np.array(matrix[:, :, i] * 255, dtype=np.uint8), h=h,
                                                     templateWindowSize=5, searchWindowSize=11)
    denoised = denoised / 255

    return denoised


def window_detection(tensor, flow, window_size, num_frames):

    vert = tensor.shape[0]
    horz = tensor.shape[1]
    m = num_frames

    norms = np.zeros(shape=(vert - window_size[0], horz - window_size[1]))

    if flow:
        # calculate sum of dense flow for the windows
        for i in np.arange(0, vert - window_size[0], 5):
            for j in np.arange(0, horz - window_size[1], 5):
                norms[int(i), int(j)] = np.sqrt(np.sum(np.square(tensor[int(i):int(i) + window_size[0],
                                                                 int(j):int(j) + window_size[1],
                                                                 :])))

    else:
        # calculate Frobenius norms for the windows on each frame and sum values
        norms = np.zeros(shape=(vert - window_size[0], horz - window_size[1]))
        for i in np.arange(0, vert - window_size[0], window_size[0]/4):
            for j in np.arange(0, horz - window_size[1], window_size[1]/4):
                norms[int(i), int(j)] = np.sqrt(np.sum(np.square(tensor[int(i):int(i) + window_size[0],
                                                                 int(j):int(j) + window_size[1],
                                                                 :])))

    # get index for window with maximum Frobenius norm
    window_index = np.unravel_index(np.argmax(norms), dims=norms.shape)

    # create mask for window index
    mask = np.zeros(shape=(vert, horz, m))
    hgt_ind, wdh_ind = window_index
    mask[hgt_ind:(hgt_ind + window_size[0]), wdh_ind:(wdh_ind + window_size[1]), :] = 1
    # mask[hgt_ind:(hgt_ind + window_size[0]), wdh_ind:(wdh_ind + window_size[1]), :] = 1
    return mask
