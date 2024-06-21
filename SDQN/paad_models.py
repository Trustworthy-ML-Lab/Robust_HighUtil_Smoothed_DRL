import torch
import torch.nn as nn
import torch.autograd as autograd
import random
import numpy as np
import sys
import torch.nn.functional as F
import math
from scipy.stats import norm
from tqdm import tqdm
import time
import torch.nn.init as init

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args,
                                                                                                                **kwargs)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class QNetwork(nn.Module):
    def __init__(self, name, input_shape, num_actions, width, m, sigma, denoise):
        super(QNetwork, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions

        self.m = m
        self.sigma = sigma
        self.denoise = denoise

        if name == 'DQN':
            self.features = nn.Sequential(
                nn.Linear(input_shape[0], 128 * width),
                nn.ReLU(),
                nn.Linear(128 * width, 128 * width),
                nn.ReLU(),
                nn.Linear(128 * width, num_actions)
            )
        elif name == 'CnnDQN':
            self.features = nn.Sequential(
                nn.Conv2d(input_shape[0], 32 * width, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32 * width, 64 * width, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64 * width, 64 * width, kernel_size=3, stride=1),
                nn.ReLU(),
                Flatten(),
                nn.Linear(3136 * width, 512 * width),
                nn.ReLU(),
                nn.Linear(512 * width, self.num_actions)
            )
        elif name == 'DuelingCnnDQN':
            self.features = DuelingCnnDQN(input_shape, self.num_actions, width)
        else:
            raise NotImplementedError('{} network structure not implemented.'.format(name))

        if self.sigma != 0.0:
            self.denoiser = Denoiser(input_shape)


    def init_record_list(self):
        self.R_list = []
        self.tot_steps = 0
        self.tot_cert = 0

    def noised_forward(self, x):
        if self.sigma != 0.0:
            x = x + torch.FloatTensor(*self.input_shape).normal_(0, self.sigma).cuda()
        if self.denoise:
            im = self.denoiser(x)
        else:
            im = x
        value = self.features(im)
        return value

    def forward(self, x):
        if self.denoise:
            im = self.denoiser(x)
        else:
            im = x
        value = self.features(im)
        return value

    def combine(self, x, dim_begin, dim_end):
        combined_shape = list(x.shape[:dim_begin]) + [-1] + list(x.shape[dim_end:])
        return x.view(combined_shape)

    def act(self, state, epsilon=0, perturb=-1, return_q=False, cert=False):
        if torch.sum(torch.isnan(state)):
            print('err', state)
            raise NotImplementedError()

        # step 1. rand smoothing


        with torch.no_grad():
            if self.sigma != 0.0:
                self.noise_list = [torch.FloatTensor(*self.input_shape).normal_(0, self.sigma).cuda() for _ in range(self.m)]
            else:
                self.noise_list = [torch.zeros(*self.input_shape).cuda() for _ in range(self.m)]

            group_size = 2000
            if self.m <= group_size:
                s_noise = torch.stack([state + self.noise_list[i] for i in range(self.m)])
                q_samples = self.forward(self.combine(s_noise, 0, 2))
                q_samples = q_samples.view([self.m, q_samples.shape[0] // self.m] + list(q_samples.shape[1:]))
                q_argmax = torch.argmax(q_samples, dim=-1, keepdim=True)
                q_samples = torch.zeros_like(q_samples).scatter_(-1, q_argmax, 1.)
                q_value = torch.mean(q_samples.float(), dim=0)
            else:
                q_value = []
                for i in range(self.m // group_size):
                    s_noise = torch.stack(
                        [state + self.noise_list[j] for j in range(i * group_size, (i + 1) * group_size)])
                    q_samples = self.forward(self.combine(s_noise, 0, 2))
                    q_samples = q_samples.view(
                        [group_size, q_samples.shape[0] // group_size] + list(q_samples.shape[1:]))
                    q_value.append(q_samples)
                q_value = torch.stack(q_value)
                q_argmax = torch.argmax(self.combine(q_value, 0, 2), dim=-1, keepdim=True)
                q_samples = torch.zeros_like(self.combine(q_value, 0, 2)).scatter_(-1, q_argmax, 1.)
                q_value = torch.mean(q_samples.float(), dim=0)



        action = q_value.max(1)[1].data.cpu().numpy()
        mask = np.random.choice(np.arange(0, 2), p=[1 - epsilon, epsilon])
        action = (1 - mask) * action + mask * np.random.randint(self.num_actions, size=state.size()[0])

        if return_q:
            return action, q_value
        return action


class DuelingCnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions, width=1):
        super(DuelingCnnDQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.cnn = nn.Sequential(
            nn.Conv2d(input_shape[0], 32 * width, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32 * width, 64 * width, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64 * width, 64 * width, kernel_size=3, stride=1),
            nn.ReLU(),
            Flatten(),
        )

        self.advantage = nn.Sequential(
            nn.Linear(3136 * width, 512 * width),
            nn.ReLU(),
            nn.Linear(512 * width, self.num_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(3136 * width, 512 * width),
            nn.ReLU(),
            nn.Linear(512 * width, 1)
        )

    def forward(self, x):
        cnn = self.cnn(x)
        advantage = self.advantage(cnn)
        value = self.value(cnn)
        return value + advantage - torch.sum(advantage, dim=1, keepdim=True) / self.num_actions


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

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y - out


def model_setup(env_id, env, use_cuda, dueling=False, model_width=1, m=1, sigma=0.1, denoise=True):
    if "NoFrameskip" not in env_id:
        net_name = 'DQN'
    else:
        if not dueling:
            net_name = 'CnnDQN'
        else:
            net_name = 'DuelingCnnDQN'
    model = QNetwork(net_name, env.observation_space.shape, env.action_space.n, model_width, m, sigma, denoise)
    if use_cuda:
        model = model.cuda()
    return model
