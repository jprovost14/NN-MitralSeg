import torch
import torch.nn as nn
import numpy as np


class NNMF(nn.Module):
    def __init__(self, matrix2d, d, dprime):
        super(NNMF, self).__init__()

        self.input_size = 2*d + dprime
        self.num_pixels = matrix2d.shape[0]
        self.num_frames = matrix2d.shape[1]

        self.dprime = dprime

        if d != 0:
            self.U = nn.Embedding(self.num_pixels, d)
            self.V = nn.Embedding(self.num_frames, d)
            self.cat = True
        else:
            self.cat = False

        self.Uprime_1 = nn.Embedding(self.num_pixels, dprime)
        self.Uprime_2 = nn.Embedding(self.num_pixels, dprime)
        self.Vprime_1 = nn.Embedding(self.num_frames, dprime)
        self.Vprime_2 = nn.Embedding(self.num_frames, dprime)

        self.ReLU = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()

        self.mlp_s = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.input_size, 1),
            nn.ReLU(),
            nn.Linear(1, 1),
            nn.Sigmoid()
        )

    def init_params(self, NMF=None):

        if NMF is None:
            def init_weights(m):
                if type(m) == nn.Sequential:
                    try:
                        nn.init.xavier_uniform_(m.weight.data, gain=4)
                    except:
                        pass
                    try:
                        nn.init.normal_(m.bias, mean=0.0, std=0.02)
                    except:
                        pass

                elif type(m) == nn.Embedding:
                    if NMF is None:
                        try:
                            # nn.init.xavier_uniform_(m.weight.data, gain=2)
                            # nn.init.xavier_normal_(m.weight.data, gain=3)
                            nn.init.normal_(m.weight.data, mean=0.5, std=0.01)
                        except:
                            pass

        else:
            def init_weights(m):
                if type(m) == nn.Sequential:
                    try:
                        nn.init.xavier_uniform_(m.weight.data, gain=4)
                    except:
                        pass
                    try:
                        nn.init.normal_(m.bias, mean=0.0, std=0.02)
                    except:
                        pass

                elif type(m) == nn.Embedding:
                    if NMF is None:
                        try:
                            nn.init.xavier_uniform_(m.weight.data, gain=2)
                        except:
                            pass

            if self.dprime == 1:
                W1 = torch.from_numpy(np.expand_dims(NMF[0][:, 0], 1)).float()
                W2 = torch.from_numpy(np.expand_dims(NMF[0][:, 1], 1)).float()
                H1 = torch.from_numpy(np.expand_dims(NMF[1][0, :], 1)).float()
                H2 = torch.from_numpy(np.expand_dims(NMF[1][1, :], 1)).float()

            else:
                noise_W1 = np.random.normal(0.5, 0.01, size=(self.num_pixels, self.dprime-1))
                noise_W2 = np.random.normal(0.5, 0.01, size=(self.num_pixels, self.dprime-1))
                noise_H1 = np.random.normal(0.5, 0.01, size=(self.num_frames, self.dprime-1))
                noise_H2 = np.random.normal(0.5, 0.01, size=(self.num_frames, self.dprime-1))

                W1 = torch.from_numpy(np.concatenate((np.expand_dims(NMF[0][:, 0], 1), noise_W1), 1)).float()
                W2 = torch.from_numpy(np.concatenate((np.expand_dims(NMF[0][:, 1], 1), noise_W2), 1)).float()
                H1 = torch.from_numpy(np.concatenate((np.expand_dims(NMF[1][0, :], 1), noise_H1), 1)).float()
                H2 = torch.from_numpy(np.concatenate((np.expand_dims(NMF[1][1, :], 1), noise_H2), 1)).float()

            self.Uprime_1 = nn.Embedding.from_pretrained(W1, freeze=False)
            self.Uprime_2 = nn.Embedding.from_pretrained(W2, freeze=False)
            self.Vprime_1 = nn.Embedding.from_pretrained(H1, freeze=False)
            self.Vprime_2 = nn.Embedding.from_pretrained(H2, freeze=False)

        self.apply(init_weights)

    def forward(self, pixel, frame, target):
        dot_prod = torch.mul(self.ReLU(self.Uprime_1(pixel)), self.ReLU(self.Vprime_1(frame))) + \
                   torch.mul(self.ReLU(self.Uprime_2(pixel)), self.ReLU(self.Vprime_2(frame)))

        if self.cat:
            mlp_input = torch.cat((self.U(pixel), self.V(frame), dot_prod), dim=1)
        else:
            mlp_input = dot_prod
        x_out = self.mlp(mlp_input)
        s_input = target - x_out
        s_out = self.mlp_s(s_input)

        del mlp_input
        del dot_prod

        return x_out, s_out
