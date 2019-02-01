import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from echos import *
import os
import numpy as np
from source.segment.nnmf import NNMF
import source.segment.utils as utils
from sklearn.decomposition import NMF


class SegNNMF:
    def __init__(self, matrix3d, sparsity_coef, beta, epochs, learning_rate, d, dprime, batchsize, num_workers,
                 save_loc, device, nmf_init):
        self.matrix3d = matrix3d/255
        self.vert = matrix3d.shape[0]  # length of image width
        self.horz = matrix3d.shape[1]  # length of image height
        self.time = matrix3d.shape[2]  # length of video
        self.n = self.horz * self.vert  # flattened image dimension
        self.d = d
        self.dprime = dprime
        self.matrix2d = self.tensor_to_matrix(self.matrix3d)
        self.nnmf = NNMF(self.matrix2d, d, dprime)
        self.epochs = epochs
        self.sparsity_coef = sparsity_coef
        self.beta = beta
        self.lr = learning_rate
        self.x_hat = torch.empty_like(torch.from_numpy(self.matrix2d), dtype=torch.float32)
        self.s = torch.empty_like(torch.from_numpy(self.matrix2d), dtype=torch.float32)
        self.batchsize = batchsize
        self.num_workers = num_workers
        self.save_loc = save_loc
        self.device = device
        self.nmf_init = nmf_init

    def tensor_to_matrix(self, matrix):
        # bring time dimension to front and flatten
        matrix2d = np.reshape(matrix, (self.n, self.time))
        return matrix2d

    @staticmethod
    def matrix_to_pixel_frame_target(matrix):
        target_vales = matrix.flatten()
        pixel_ind = np.repeat(np.arange(matrix.shape[0]), matrix.shape[1])
        frame_ind = np.tile(np.arange(matrix.shape[1]), matrix.shape[0])
        index_mat = np.vstack((pixel_ind, frame_ind, target_vales)).T
        return index_mat

    def load_dataset(self, ind_mat, batch_size):
        dataset = utils.MyDataset(ind_mat)
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                  shuffle=True, num_workers=self.num_workers,
                                                  drop_last=False)

        return trainloader

    def l2_norm(self, vector):
        norm = torch.sqrt(torch.sum(torch.pow(vector, 2))) / self.batchsize
        return norm

    def mse_loss(self, target, x_out):
        loss = nn.functional.mse_loss(x_out, target)
        return loss

    def lat_loss(self, pixel, frame):
        if self.d != 0:
            loss = self.beta * (self.l2_norm(self.nnmf.U(pixel)) +
                                self.l2_norm(torch.cat((self.nnmf.Uprime_1(pixel), self.nnmf.Uprime_2(pixel)))) +
                                self.l2_norm(self.nnmf.V(frame)) +
                                self.l2_norm(torch.cat((self.nnmf.Vprime_1(frame), self.nnmf.Vprime_2(frame)))))
        else:
            loss = self.beta * (self.l2_norm(torch.cat((self.nnmf.Uprime_1(pixel), self.nnmf.Uprime_2(pixel)))) +
                                self.l2_norm(torch.cat((self.nnmf.Vprime_1(frame), self.nnmf.Vprime_2(frame)))))
        return loss

    def l1_loss(self, s_out):
        loss = self.sparsity_coef * torch.sum(torch.abs(s_out)) / self.batchsize
        return loss

    @staticmethod
    def get_free_gpu():
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
        return np.argmax(memory_available)

    def train(self):
        # initialize epochs
        ep = 0

        # create data loader
        print('getting ind matrix')
        ind_mat = self.matrix_to_pixel_frame_target(self.matrix2d)
        print('loading dataset')
        trainloader = self.load_dataset(ind_mat, batch_size=self.batchsize)

        # initialize network parameters and send to gpu
        if self.nmf_init:
            model = NMF(n_components=2, init='random', random_state=0, max_iter=200, tol=0.0001)
            W = model.fit_transform(self.matrix2d)
            H = model.components_
            self.nnmf.init_params(NMF=(W, H))
        else:
            self.nnmf.init_params()

        self.nnmf.to(self.device)

        # optimizers for mlp and latent features

        if self.d != 0:
            latent_params = list(self.nnmf.U.parameters()) + list(self.nnmf.Uprime_1.parameters()) + \
                            list(self.nnmf.Uprime_2.parameters()) + list(self.nnmf.V.parameters()) + \
                            list(self.nnmf.Vprime_1.parameters()) + list(self.nnmf.Vprime_2.parameters())
        else:
            latent_params = list(self.nnmf.Uprime_1.parameters()) + list(self.nnmf.Uprime_2.parameters()) + \
                            list(self.nnmf.Vprime_1.parameters()) + list(self.nnmf.Vprime_2.parameters())  # + \

        latent_opt = optim.RMSprop(latent_params, lr=self.lr)
        # latent_opt = optim.Adam(latent_params, lr=self.lr)

        mlp_opt = optim.RMSprop(self.nnmf.mlp.parameters(), lr=self.lr)
        # mlp_opt = optim.Adam(self.nnmf.mlp.parameters(), lr=self.lr)

        mlp_s_opt = optim.RMSprop(self.nnmf.mlp_s.parameters(), lr=self.lr)
        # mlp_s_opt = optim.Adam(self.nnmf.mlp_s.parameters(), lr=self.lr)

        print('beginning training')

        while ep < self.epochs:

            for batch_id, batch in enumerate(trainloader, 0):

                # get data from batch
                pixel, frame, target = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])

                # send x_hat and s to gpu
                self.x_hat = self.x_hat.to(self.device)
                self.s = self.s.to(self.device)

                # send to gpu
                pixel = pixel.to(self.device)
                frame = frame.to(self.device)
                target = target.to(self.device).float()
                target = torch.reshape(target, shape=(target.shape[0], 1))

                if ep >= 15:
                    ##################################################################
                    # train mlp_s weights
                    mlp_s_opt.zero_grad()
                    x_out, s_out = self.nnmf.forward(pixel, frame, target)

                    # calculate loss on mlp_s and update mlp_s weights
                    mse_loss = self.mse_loss(target, x_out + s_out)
                    l1_loss = self.l1_loss(s_out)
                    if ep >= 20:
                        loss_mlp = mse_loss + l1_loss
                    else:
                        loss_mlp = mse_loss
                    loss_mlp.backward()
                    mlp_s_opt.step()

                ##################################################################
                # train mlp weights
                mlp_opt.zero_grad()
                x_out, s_out = self.nnmf.forward(pixel, frame, target)

                # calculate loss on mlp and update mlp weights
                if ep >= 15:
                    mse_loss = self.mse_loss(target, x_out + s_out)
                else:
                    mse_loss = self.mse_loss(target, x_out)
                lat_loss = self.lat_loss(pixel, frame)
                loss_mlp_x = mse_loss + lat_loss
                loss_mlp_x.backward()
                mlp_opt.step()

                ##################################################################
                # train latent weights
                latent_opt.zero_grad()
                x_out, s_out = self.nnmf.forward(pixel, frame, target)

                # calculate loss on latent and update latent weights
                if ep >= 15:
                    mse_loss = self.mse_loss(target, x_out + s_out)
                else:
                    mse_loss = self.mse_loss(target, x_out)
                    l1_loss = 0
                lat_loss = self.lat_loss(pixel, frame)
                loss_latent = mse_loss + lat_loss
                loss_latent.backward()
                latent_opt.step()

                # update x_hat and s
                self.x_hat[pixel, frame] = torch.squeeze(x_out.detach())
                self.s[pixel, frame] = torch.squeeze(s_out.detach())

                # print loss
                if batch_id % 10 == 0:
                    print('[%d] MSE Loss: %.5f | Latent Loss: %.5f | L1 Loss: %.5f' %
                          (ep + 1, mse_loss, lat_loss, l1_loss))

                # save x_hat and s to directory
            dir = '/local/home/jprovost/echo/out/mitral_valve/nnmf/' + self.save_loc + '/'
            print('Saving to dir:', dir)
            try:
                os.mkdir(dir)
                torch.save(self.x_hat.cpu(), dir + 'x_hat.pt')
                torch.save(self.s.cpu(), dir + 's.pt')
                torch.save(self.nnmf.Uprime_1.weight.cpu(), dir + 'Uprime1.pt')
                torch.save(self.nnmf.Uprime_2.weight.cpu(), dir + 'Uprime2.pt')
            except:
                torch.save(self.x_hat.cpu(), dir + 'x_hat.pt')
                torch.save(self.s.cpu(), dir + 's.pt')
                torch.save(self.nnmf.Uprime_1.weight.cpu(), dir + 'Uprime1.pt')
                torch.save(self.nnmf.Uprime_2.weight.cpu(), dir + 'Uprime2.pt')

            # increase epoch
            ep += 1
