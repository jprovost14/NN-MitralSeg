import medpy.filter.smoothing as mp
from sklearn.decomposition import NMF
import os
from source.utils import *


class SegRNMF:
    def __init__(self, matrix3d, sparsity_coef, max_iter, save_loc, option='rnmf_seg',
                 window_size=(60, 80), init='nmf', rank=2, thresh1=95, thresh2=98):
        self.matrix3d = matrix3d/255
        self.sparsity_coef = sparsity_coef
        self.max_iter = max_iter
        self.option = option
        self.vert = matrix3d.shape[0]  # length of image width
        self.horz = matrix3d.shape[1]  # length of image height
        self.n = self.horz * self.vert  # flattened image dimension
        self.k = rank
        self.m = matrix3d.shape[2]  # number of frames
        self.init = init
        self.thresh1 = thresh1  # percentile for threshold
        self.thresh2 = thresh2
        self.window_size = window_size  # size of window for mitral valve
        self.window_index = None
        self.save_loc = save_loc

    def tensor_to_matrix(self):
        # bring time dimension to front and flatten
        matrix2d = np.reshape(self.matrix3d, (self.n, self.m))
        return matrix2d

    def rnmf(self, matrix2d, sparsity_coef):

        i = 0

        if self.init == 'nmf':
            model = NMF(n_components=self.k, init='random', random_state=0, max_iter=200, tol=0.0001)
            W = model.fit_transform(matrix2d)
            H = model.components_

        else:
            W = np.random.uniform(0, 1, size=(self.n, self.k))
            H = np.random.uniform(0, 1, size=(self.k, self.m))

        while i <= self.max_iter:

            W_old = W
            H_old = H
            # initialize S matrix
            S = matrix2d - (W_old @ H_old)

            # update S matrix
            S[S > sparsity_coef / 2] = S[S > sparsity_coef/2] - sparsity_coef/2
            S[S < sparsity_coef / 2] = 0

            # update W matrix
            W_new = W_old * (np.maximum(matrix2d - S, 0) @ H_old.T) / (W_old @ H_old @ H_old.T)
            nan_ind = np.isnan(W_new)
            inf_ind = np.isinf(W_new)
            W_new[nan_ind] = 0
            W_new[inf_ind] = 1
            W_new = W_new / np.linalg.norm(W_new, ord='fro', keepdims=True)

            # update H matrix
            H_new = H_old * (W_new.T @ np.maximum(matrix2d - S, 0)) / (W_new.T @ W_new @ H_old)
            nan_ind = np.isnan(H_new)
            inf_ind = np.isinf(H_new)
            H_new[nan_ind] = 0
            H_new[inf_ind] = 1

            # normalize W and H
            W = W_new
            # H = H_new
            H = H_new * np.linalg.norm(W_new, ord='fro', keepdims=True)

            # if i % 5 == 0:
            #     print(np.mean(S))
            #     print(np.mean(W @ H))

            i += 1

        return W, H, S

    def matrix_to_tensor(self, W, H, S):
        WH = np.reshape(W @ H, newshape=(self.vert, self.horz, self.m))
        S = np.reshape(S, newshape=(self.vert, self.horz, self.m))
        return WH, S

    def thresholding_fn(self, matrix, thresh, thresh_func):

        if thresh_func == 'percentile':
            # calculate threshold value for given percentile
            thresh_val = np.percentile(matrix, q=thresh)
            # assign binary values to pixels above and below threshold
            matrix[matrix < thresh_val] = matrix[matrix < thresh_val] * 0

        else:
            for i in range(self.m):
                matrix[:, :, i] = cv2.GaussianBlur(np.array(matrix[:, :, i] * 255, dtype=np.uint8), (21, 21), sigmaX=0.5, sigmaY=0.5)
                _, matrix[:, :, i] = cv2.threshold(np.array(matrix[:, :, i], dtype=np.uint8), 0, 255,
                                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return matrix

    def remove_valve(self, WH, S, mask):

        M = WH + S

        # take window of threshold S matrix
        S_prime = mask * thresholding_fn(S, thresh=self.thresh1)

        # anisotropic diffusion to connect segments
        S_aniso = np.empty_like(S_prime)
        for j in range(S_prime.shape[2]):
            S_aniso[:, :, j] = mp.anisotropic_diffusion(S_prime[:, :, j], niter=5, kappa=20)

        # remove from rnmf
        S_aniso = np.reshape(S_prime, newshape=(self.vert, self.horz, self.m))

        # M_prime = np.maximum(0, M-S_prime)
        M_prime = M - M * S_aniso

        return M, M_prime

    def get_valve(self, M, W2H2, mask):
        # take window of threshold difference between myocardium and reconstruction
        thresh = thresholding_fn(M-W2H2, thresh=self.thresh2)

        valve = thresh * mask

        valve_aniso = np.empty_like(valve)
        # anisotropic diffusion to connect segments
        for j in range(M.shape[2]):
            valve_aniso[:, :, j] = mp.anisotropic_diffusion(valve[:, :, j], niter=5, kappa=20)

        valve_aniso[valve_aniso > 0] = 1

        return valve_aniso

    def segment(self):
        # convert echo to 2D matrix
        print("Converting to 2D")
        matrix2d = self.tensor_to_matrix()

        # RNMF on 2D representation
        print("RNMF #1")
        W1, H1, S1 = self.rnmf(matrix2d, sparsity_coef=self.sparsity_coef[0])

        # threshold S
        S1 = thresholding_fn(S1, thresh=self.thresh1, )

        # convert to tensor for window detection
        print("Convert to Tensor and Window Detection")
        W1H1, S1 = self.matrix_to_tensor(W1, H1, S1)
        # animate(S1)

        # np.save('/Users/jesseprovost/Downloads/echo/out/mitral_valve/rnmf/original/4CH/17/s_full.npy', S1)

        for i in range(S1.shape[2]):
            fig, axs = plt.subplots(1, 3, sharey=True)
            # fig.subplots_adjust(wspace=0)
            axs[0].imshow(S1[:, :, i], cmap='gray')
            axs[0].axis('off')
            axs[1].imshow(S1[:, :, i+1], cmap='gray')
            axs[1].axis('off')
            axs[2].imshow(S1[:, :, i+2], cmap='gray')
            axs[2].axis('off')
            plt.show()

        if self.option == 'rnmf_seg':
            mask = window_detection(tensor=S1, flow=False, window_size=self.window_size, num_frames=self.m)
        else:
            flow = optical_flow(S1, winsize=40)
            # plt.imshow(np.sum(flow, axis=2))
            # plt.show()
            mask = window_detection(tensor=flow, flow=True, window_size=self.window_size, num_frames=self.m)
            # animate(S1 * mask)

        # remove valve from RNMF reconstruction
        print("Removing Valve")
        # animate(S1)
        M, M_prime = self.remove_valve(W1H1, S1, mask)

        # reshape to 2D for RNMF on echo without valve to get muscle motion
        print("RNMF #2")
        M_prime = np.reshape(M_prime, (self.n, self.m))
        W, H, S = self.rnmf(M_prime, sparsity_coef=self.sparsity_coef[1])

        # get myocardium motion from RNMF by converting back to tensor (video)
        print("Getting Myocardium")
        W2H2, S = self.matrix_to_tensor(W, H, S)
        myocardium = np.reshape(W2H2, newshape=(self.vert, self.horz, self.m))

        # get valve by taking difference from original reconstruction and reconstruction without valve
        print("Getting Valve")
        valve = self.get_valve(M, W2H2, mask)

        # get noise
        # print("Getting Noise")
        # noise = np.maximum(np.reshape(matrix2d,
        #                               newshape=(self.vert, self.horz, self.m)) - myocardium - valve, 0)

        if self.option == 'rnmf_opt_seg':
            dir = '/Users/jesseprovost/Downloads/echo/out/mitral_valve/rnmf/optical_flow/' + self.save_loc + '/'
        else:
            dir = '/Users/jesseprovost/Downloads/echo/out/mitral_valve/rnmf/original/' + self.save_loc + '/'
        print('Saving to dir:', dir)

        if not os.path.exists(dir):
            os.makedirs(dir)

        np.save(file=dir + 's.npy', arr=valve)
        np.save(file=dir + 'x_hat.npy', arr=myocardium)
        np.save(file=dir + 'mask.npy', arr=mask)

        return valve, myocardium, mask
