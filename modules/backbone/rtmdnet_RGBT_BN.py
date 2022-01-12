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



class MDNet_RGBT(nn.Module):
    def __init__(self, model_path=None, K=1):
        super(MDNet_RGBT, self).__init__()
        self.K = K
        self.layers = nn.Sequential(OrderedDict([
            ('conv1', nn.Sequential(nn.Conv2d(4, 96, kernel_size=7, stride=2),
                                    nn.BatchNorm2d(96),
                                    nn.ReLU(),
                                    # LRN(),
                                    nn.MaxPool2d(kernel_size=3, stride=2)
                                    )),
            ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2, dilation=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(),
                                    # LRN(),
                                    )),

            ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, dilation=3),
                                    nn.ReLU(),
                                    )),
            ('fc4', nn.Sequential(nn.Linear(512 * 3 * 3, 512),
                                  nn.ReLU())),
            ('fc5', nn.Sequential(nn.Dropout(0.5),
                                  nn.Linear(512, 512),
                                  nn.ReLU()))]))

        self.branches = nn.ModuleList([nn.Sequential(nn.Dropout(0.5),
                                                     nn.Linear(512, 2)) for _ in range(K)])

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

    def build_param_dict(self):
        self.params = OrderedDict()
        for name, module in self.layers.named_children():
            append_params(self.params, module, name)
        for k, module in enumerate(self.branches):
            append_params(self.params, module, 'fc6_%d' % k)

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

    def forward(self, x, k=0, in_layer='conv1', out_layer='fc6'):

        run = False
        for name, module in self.layers.named_children():
            if name == in_layer:
                run = True
            if run:
                x = module(x)
                if name == out_layer:
                    return x

        x = self.branches[k](x)
        if out_layer == 'fc6':
            return x
        elif out_layer == 'fc6_softmax':
            return F.softmax(x)

    def load_model(self, model_path):
        states = torch.load(model_path)
        shared_layers = states['shared_layers']
        try:
            self.layers.load_state_dict(shared_layers)
        except:
            self.layers.load_state_dict(shared_layers, strict=False)
            print('Missing key(s) in state_dict, already set strict=False')
        print('load .pth model finish......')

    def load_mat_model(self, matfile):
        mat = scipy.io.loadmat(matfile)
        mat_layers = list(mat['layers'])[0]

        # copy conv weights
        for i in range(3):
            weight, bias = mat_layers[i * 4]['weights'].item()[0]
            if i == 0:
                self.layers[i][0].weight.data = torch.cat((
                    torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))[:, 0, :, :].view(
                        weight.shape[3], 1, weight.shape[0], weight.shape[1]),
                    torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))), 1)
                self.layers[i][0].bias.data = torch.from_numpy(bias[:, 0])
            else:
                self.layers[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
                self.layers[i][0].bias.data = torch.from_numpy(bias[:, 0])
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
