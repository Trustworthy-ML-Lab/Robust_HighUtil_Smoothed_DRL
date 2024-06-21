import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import random
import numpy as np
import sys
import math
import torch.nn.init as init


USE_CUDA = torch.cuda.is_available()


class Denoiser(nn.Module):
    def __init__(self, input_shape):
        super(Denoiser, self).__init__()
        self.input_shape = input_shape


        channels = 64
        layers = []

        layers.append(
            nn.Conv2d(in_channels=input_shape[0], out_channels=channels, kernel_size=3, padding=1,
                      bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(15):
            layers.append(
                nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1,
                          bias=False))
            layers.append(nn.BatchNorm2d(channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(in_channels=channels, out_channels=input_shape[0], kernel_size=3, padding=1,
                      bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        lastcnn = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                lastcnn = m
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
        init.constant_(lastcnn.weight, 0)

    def forward(self, x, sigma_schedule=0.0):
        if sigma_schedule != 0.0:
            x_noise = x + torch.FloatTensor(*self.input_shape).normal_(0, sigma_schedule).cuda()
            out = self.dncnn(x_noise)
            return x_noise - out
        else:
            y = x
            out = self.dncnn(x)
            return y - out


def denoiser_setup(env, use_cuda):
    model = Denoiser(env.observation_space.shape)
    if use_cuda:
        model = model.cuda()
    return model
