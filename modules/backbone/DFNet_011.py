import os
import scipy.io
import numpy as np
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch

import time

import sys

sys.path.insert(0, '../prroi_pool')
from modules.prroi_pool import PrRoIPool2D

import cv2
import torch


def append_params(params, module, prefix):
    for child in module.children():
        for k, p in child.named_parameters():
            if p is None:
                continue

            if isinstance(child, nn.BatchNorm2d):
                name = prefix + '_bn_' + k
            else:
                name = prefix + '_' + k

            if name not in params:
                params[name] = p
            else:
                raise RuntimeError("Duplicated param name: %s" % name)


class attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()
        assert temperature % 3 == 1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes != 3:
            hidden_planes = int(in_planes * ratios)
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=False)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature != 1:
            self.temperature -= 1
            # print('Change temperature to:', str(self.temperature))

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x / self.temperature, 1)


class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, K=2, temperature=31, init_weight=True):
        super(Dynamic_conv2d, self).__init__()
        assert in_planes % groups == 0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention2d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(torch.Tensor(K - 1, out_planes, in_planes // groups, kernel_size, kernel_size),
                                   requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(K - 1, out_planes))
            nn.init.zeros_(self.bias)
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.K - 1):
            nn.init.kaiming_uniform_(self.weight[i])

    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x, common_weight, common_bias):  # 将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        softmax_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x.contiguous().view(1, -1, height, width)  # 变化成一个维度进行组卷积
        weight = torch.cat((common_weight, self.weight), dim=0).view(self.K, -1)
        # self.weight.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.in_planes, self.kernel_size,
                                                                    self.kernel_size)
        if self.bias is not None:
            two_bias = torch.cat((common_bias, self.bias), dim=0)
            aggregate_bias = torch.mm(softmax_attention, two_bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output


class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=0.0001, beta=0.75, ACROSS_CHANNELS=False):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if self.ACROSS_CHANNELS:
            self.average = nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                                        stride=1,
                                        padding=(int((local_size - 1.0) / 2), 0, 0))
        else:
            self.average = nn.AvgPool2d(kernel_size=local_size,
                                        stride=1,
                                        padding=int((local_size - 1.0) / 2))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(2.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(2.0).pow(self.beta)
        x = x.div(div)
        return x


class DFNet011(nn.Module):
    def __init__(self, model_path=None, K=1):
        super(DFNet011, self).__init__()
        self.K = K

        self.conv2weight = nn.Parameter(torch.Tensor(1, 256, 96, 5, 5), requires_grad=True)
        self.conv2bias = nn.Parameter(torch.Tensor(1, 256), requires_grad=True)
        self.conv3weight = nn.Parameter(torch.Tensor(1, 512, 256, 3, 3), requires_grad=True)
        self.conv3bias = nn.Parameter(torch.Tensor(1, 512), requires_grad=True)

        self.RGB_layers = nn.Sequential(nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                                      nn.ReLU(),
                                                      LRN(),
                                                      nn.MaxPool2d(kernel_size=3, stride=2)
                                                      ),
                                        nn.ModuleList([Dynamic_conv2d(96, 256, kernel_size=5, stride=2, dilation=1),
                                                       nn.ReLU(),
                                                       LRN(),
                                                       ]),
                                        nn.ModuleList([Dynamic_conv2d(256, 512, kernel_size=3, stride=1, dilation=3),
                                                       nn.ReLU(),
                                                       ]))
        self.T_layers = nn.Sequential(nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                                    nn.ReLU(),
                                                    LRN(),
                                                    nn.MaxPool2d(kernel_size=3, stride=2)
                                                    ),
                                      nn.ModuleList([Dynamic_conv2d(96, 256, kernel_size=5, stride=2, dilation=1),
                                                     nn.ReLU(),
                                                     LRN(),
                                                     ]),
                                      nn.ModuleList([Dynamic_conv2d(256, 512, kernel_size=3, stride=1, dilation=3),
                                                     nn.ReLU(),
                                                     ]))
        self.layers = nn.Sequential(OrderedDict([('conv1', nn.Module()),
                                                 ('conv2', nn.Module()),
                                                 ('conv3', nn.Module()),
                                                 ('fc4', nn.Sequential(nn.Linear(1024 * 3 * 3, 512),
                                                                       nn.ReLU())),
                                                 ('fc5', nn.Sequential(nn.Dropout(0.5),
                                                                       nn.Linear(512, 512),
                                                                       nn.ReLU()))]))

        self.branches = nn.ModuleList([nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 2)) for _ in range(K)])

        self.roi_pool_model = PrRoIPool2D(3, 3, 1. / 8)

        self.receptive_field = 75.  # it is receptive fieald that a element of feat_map covers. feat_map is bottom layer of ROI_align_layer

        if model_path is not None:
            if os.path.splitext(model_path)[1] == '.pth':
                self.load_model(model_path)
            elif os.path.splitext(model_path)[1] == '.mat':
                self.load_mat_model(model_path)
            else:
                raise RuntimeError("Unkown model format: %s" % model_path)
        self.build_param_dict()

    def update_temperature(self):
        for m in self.modules():
            if isinstance(m, Dynamic_conv2d):
                m.update_temperature()

    def build_param_dict(self):
        self.params = OrderedDict()
        for name, module in self.T_layers.named_children():
            append_params(self.params, module, 'conv_T_layers' + name)
        for name, module in self.RGB_layers.named_children():
            append_params(self.params, module, 'conv_RGB_layers' + name)
        for name, module in self.layers.named_children():
            append_params(self.params, module, name)
        for k, module in enumerate(self.branches):
            append_params(self.params, module, 'fc6_%d' % k)
        self.params['conv2weight'] = self.conv2weight
        self.params['conv2bias'] = self.conv2bias
        self.params['conv3weight'] = self.conv3weight
        self.params['conv3bias'] = self.conv3bias

    def set_learnable_params(self, layers):
        for k, p in self.params.items():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.params.items():
            if p.requires_grad:
                params[k] = p
        return params

    def forward(self, R=None, T=None, feat=None, k=0, in_layer='conv1', out_layer='fc6'):
        run = False
        for nameid, (name, module) in enumerate(self.layers.named_children()):
            if name == in_layer:
                run = True
            if run:
                if name == 'conv1':
                    featR = self.RGB_layers[nameid](R)
                    featT = self.T_layers[nameid](T)
                elif name == 'conv2':
                    featR = self.RGB_layers[nameid][0](featR, self.conv2weight, self.conv2bias)
                    featR = self.RGB_layers[nameid][1](featR)
                    featR = self.RGB_layers[nameid][2](featR)
                    featT = self.T_layers[nameid][0](featT, self.conv2weight, self.conv2bias)
                    featT = self.T_layers[nameid][1](featT)
                    featT = self.T_layers[nameid][2](featT)
                elif name == 'conv3':
                    featR = self.RGB_layers[nameid][0](featR, self.conv3weight, self.conv3bias)
                    featR = self.RGB_layers[nameid][1](featR)
                    featT = self.T_layers[nameid][0](featT, self.conv3weight, self.conv3bias)
                    featT = self.T_layers[nameid][1](featT)
                    feat = torch.cat((featR, featT), 1)
                elif name == 'fc4':
                    feat = module(feat)
                elif name == 'fc5':
                    feat = module(feat)
                if name == out_layer:
                    return feat

        feat = self.branches[k](feat)
        if out_layer == 'fc6':
            return feat
        elif out_layer == 'fc6_softmax':
            return F.softmax(feat)

    def load_model(self, model_path):
        states = torch.load(model_path)
        try:
            self.layers.load_state_dict(states['shared_layers'])
        except:
            self.layers.load_state_dict(states['shared_layers'], strict=False)
            print('Missing key(s) in shared_layers, already set strict=False')
        try:
            self.RGB_layers.load_state_dict(states['RGB_layers'])
        except:
            self.RGB_layers.load_state_dict(states['RGB_layers'], strict=False)
            print('Missing key(s) in RGB_layers, already set strict=False')
        try:
            self.T_layers.load_state_dict(states['T_layers'])
        except:
            self.T_layers.load_state_dict(states['T_layers'], strict=False)
            print('Missing key(s) in T_layers, already set strict=False')
        self.conv2weight.data = states['conv2weight']
        self.conv2bias.data = states['conv2bias']
        self.conv3weight.data = states['conv3weight']
        self.conv3bias.data = states['conv3bias']
        print('load .pth model finish......')

    def load_mat_model(self, matfile):
        mat = scipy.io.loadmat(matfile)
        mat_layers = list(mat['layers'])[0]

        # copy conv weights
        for i in range(3):
            weight, bias = mat_layers[i * 4]['weights'].item()[0]
            if i != 0:
                self.RGB_layers[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1))).unsqueeze(0)
                self.RGB_layers[i][0].bias.data = torch.from_numpy(bias[:, 0]).unsqueeze(0)
                self.T_layers[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1))).unsqueeze(0)
                self.T_layers[i][0].bias.data = torch.from_numpy(bias[:, 0]).unsqueeze(0)
                getattr(self, 'conv' + str(i + 1) + 'weight').data = torch.from_numpy(
                    np.transpose(weight, (3, 2, 0, 1))).unsqueeze(0)
                getattr(self, 'conv' + str(i + 1) + 'bias').data = torch.from_numpy(bias[:, 0]).unsqueeze(0)
            else:
                self.RGB_layers[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
                self.RGB_layers[i][0].bias.data = torch.from_numpy(bias[:, 0])
                self.T_layers[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
                self.T_layers[i][0].bias.data = torch.from_numpy(bias[:, 0])
        print('load .mat model finish......')

    def trainSpatialTransform(self, image, bb):

        return


class BinaryLoss(nn.Module):
    def __init__(self):
        super(BinaryLoss, self).__init__()

    def forward(self, pos_score, neg_score):
        pos_loss = -F.log_softmax(pos_score, dim=1)[:, 1]
        neg_loss = -F.log_softmax(neg_score, dim=1)[:, 0]

        loss = (pos_loss.sum() + neg_loss.sum()) / (pos_loss.size(0) + neg_loss.size(0))
        return loss


class Accuracy:
    def __call__(self, pos_score, neg_score):
        pos_correct = (pos_score[:, 1] > pos_score[:, 0]).sum().float()
        neg_correct = (neg_score[:, 1] < neg_score[:, 0]).sum().float()

        pos_acc = pos_correct / (pos_score.size(0) + 1e-8)
        neg_acc = neg_correct / (neg_score.size(0) + 1e-8)

        return pos_acc.item(), neg_acc.item()


class Precision:
    def __call__(self, pos_score, neg_score):
        scores = torch.cat((pos_score[:, 1], neg_score[:, 1]), 0)
        topk = torch.topk(scores, pos_score.size(0))[1]
        prec = (topk < pos_score.size(0)).float().sum() / (pos_score.size(0) + 1e-8)

        return prec.item()


if __name__ == "__main__":
    model = DFNet011('/home/pjc/MyProgram/RT-MDNet/models/imagenet-vgg-m.mat')
    print(model.conv2weight.shape)
    print(model.conv2bias.shape)
    print(model.conv3weight.shape)
    print(model.conv3bias.shape)
    x = torch.rand(1, 3, 107, 107)
    model.cuda()
    y = model(x.cuda(), x.cuda(), out_layer='conv3')
    z = model.roi_pool_model(y, torch.tensor([[0, 0, 0, 10, 10], ]).float().cuda())
    z = z.view(z.size(0), -1)
    z = model(feat=z, in_layer='fc4', out_layer='fc6')
