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
    if len(list(module.children())) != 0:
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
    else:
        for k, p in module.named_parameters():
            if p is None:
                continue

            if isinstance(module, nn.BatchNorm2d):
                name = prefix + '_bn_' + k
            else:
                name = prefix + '_' + k

            if name not in params:
                params[name] = p
            else:
                raise RuntimeError("Duplicated param name: %s" % name)


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


class IVFuseNet(nn.Module):
    def __init__(self, model_path=None, K=1):
        super(IVFuseNet, self).__init__()
        self.K = K
        self.RGB_layers = nn.Sequential(nn.Conv2d(3, 72, kernel_size=7, stride=2),
                                        nn.Conv2d(96, 128, kernel_size=5, stride=2, dilation=1),
                                        nn.Conv2d(256, 128, kernel_size=3, stride=1, dilation=3))
        self.T_layers = nn.Sequential(nn.Conv2d(3, 72, kernel_size=7, stride=2),
                                      nn.Conv2d(96, 128, kernel_size=5, stride=2, dilation=1),
                                      nn.Conv2d(256, 128, kernel_size=3, stride=1, dilation=3))
        self.layers = nn.Sequential(OrderedDict([
            ('conv1', nn.Sequential(nn.Conv2d(3, 24, kernel_size=7, stride=2),
                                    nn.ReLU(),
                                    LRN(),
                                    nn.MaxPool2d(kernel_size=3, stride=2)
                                    )),
            ('conv2', nn.Sequential(nn.Conv2d(96, 128, kernel_size=5, stride=2, dilation=1),
                                    nn.ReLU(),
                                    LRN(),
                                    )),
            ('conv3', nn.Sequential(nn.Conv2d(256, 384, kernel_size=3, stride=1, dilation=3),
                                    nn.ReLU(),
                                    )),
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
        # TODO fix in train and tracker for "model(x, k)" and "model(R=, T=, k=)"
        run = False
        for nameid, (name, module) in enumerate(self.layers.named_children()):
            if name == in_layer:
                run = True
            if run:
                if name == 'conv1':
                    feat_R = self.RGB_layers[nameid](R)
                    feat_T = self.T_layers[nameid](T)
                    feat_MR = module[0](R)
                    feat_MT = module[0](T)
                    featR = torch.cat((feat_R, feat_MR), 1)
                    featT = torch.cat((feat_T, feat_MT), 1)
                    featR = module[1](featR)
                    featR = module[2](featR)
                    featR = module[3](featR)
                    featT = module[1](featT)
                    featT = module[2](featT)
                    featT = module[3](featT)
                elif name == 'conv2':
                    feat_R = self.RGB_layers[nameid](featR)
                    feat_T = self.T_layers[nameid](featT)
                    feat_MR = module[0](featR)
                    feat_MT = module[0](featT)
                    featR = torch.cat((feat_R, feat_MR), 1)
                    featT = torch.cat((feat_T, feat_MT), 1)
                    featR = module[1](featR)
                    featR = module[2](featR)
                    featT = module[1](featT)
                    featT = module[2](featT)
                elif name == 'conv3':
                    feat_R = self.RGB_layers[nameid](featR)
                    feat_T = self.T_layers[nameid](featT)
                    feat_MR = module[0](featR)
                    feat_MT = module[0](featT)
                    featR = torch.cat((feat_R, feat_MR), 1)
                    featT = torch.cat((feat_T, feat_MT), 1)
                    featR = module[1](featR)
                    featT = module[1](featT)
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
        print('load .pth model finish......')

    def load_mat_model(self, matfile):
        mat = scipy.io.loadmat(matfile)
        mat_layers = list(mat['layers'])[0]

        # copy conv weights
        for i in range(3):
            if i == 0:
                weight, bias = mat_layers[i * 4]['weights'].item()[0]
                self.RGB_layers[i].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))[:72, :, :, :]
                self.RGB_layers[i].bias.data = torch.from_numpy(bias[:, 0])[:72]
                self.T_layers[i].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))[:72, :, :, :]
                self.T_layers[i].bias.data = torch.from_numpy(bias[:, 0])[:72]
                self.layers[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))[72:, :, :, :]
                self.layers[i][0].bias.data = torch.from_numpy(bias[:, 0])[72:]
            else:
                weight, bias = mat_layers[i * 4]['weights'].item()[0]
                self.RGB_layers[i].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))[:128, :, :, :]
                self.RGB_layers[i].bias.data = torch.from_numpy(bias[:, 0])[:128]
                self.T_layers[i].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))[:128, :, :, :]
                self.T_layers[i].bias.data = torch.from_numpy(bias[:, 0])[:128]
                self.layers[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))[128:, :, :, :]
                self.layers[i][0].bias.data = torch.from_numpy(bias[:, 0])[128:]
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
    model = IVFuseNet('/home/pjc/MyProgram/RT-MDNet/models/imagenet-vgg-m.mat')
    print(model.RGB_layers[0].weight.shape)
    print(model.RGB_layers[0].bias.shape)
    print(model.T_layers[0].weight.shape)
    print(model.T_layers[0].bias.shape)
    print(model.layers[0][0].weight.shape)
    print(model.layers[0][0].bias.shape)
    print(model.RGB_layers[1].weight.shape)
    print(model.RGB_layers[1].bias.shape)
    print(model.T_layers[1].weight.shape)
    print(model.T_layers[1].bias.shape)
    print(model.layers[1][0].weight.shape)
    print(model.layers[1][0].bias.shape)
    print(model.RGB_layers[2].weight.shape)
    print(model.RGB_layers[2].bias.shape)
    print(model.T_layers[2].weight.shape)
    print(model.T_layers[2].bias.shape)
    print(model.layers[2][0].weight.shape)
    print(model.layers[2][0].bias.shape)
    x = torch.rand(1, 3, 107, 107)
    model.cuda()
    y = model(x.cuda(), x.cuda(), out_layer='conv3')
    z = model.roi_pool_model(y, torch.tensor([[0, 0, 0, 10, 10], ]).float().cuda())
    z = z.view(z.size(0), -1)
    z = model(feat=z, in_layer='fc4', out_layer='fc6')
