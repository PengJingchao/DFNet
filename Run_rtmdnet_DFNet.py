import os
from os.path import join, isdir
from tracker_rtmdnet_DFNet import *  # todo change tracker
import numpy as np
import random
import argparse

import pickle

import math


def init_seed(seed=None):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def genConfig(seq_path, set_type, result_home):
    if set_type == 'RGBT234':
        rgb_img_dir = os.path.join(seq_path, 'visible')
        t_img_dir = os.path.join(seq_path, 'infrared')
    elif set_type == 'GTOT':
        rgb_img_dir = os.path.join(seq_path, 'v')
        t_img_dir = os.path.join(seq_path, 'i')

    gt_path = os.path.join(seq_path, 'init.txt')
    rgb_img_list = os.listdir(rgb_img_dir)
    rgb_img_list.sort()
    rgb_img_list = [os.path.join(rgb_img_dir, x) for x in rgb_img_list]
    t_img_list = os.listdir(t_img_dir)
    t_img_list.sort()
    t_img_list = [os.path.join(t_img_dir, x) for x in t_img_list]
    with open(gt_path) as f:
        gt = np.loadtxt((x.replace(',', ' ') for x in f))
    init_bbox = gt[0]

    if not os.path.exists(result_home):
        os.makedirs(result_home)
    result_path = os.path.join(result_home, seq_path.split('/')[-1] + '.txt')
    return rgb_img_list, t_img_list, init_bbox, gt, result_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-set_type", default='GTOT')
    parser.add_argument("-model_path", default='./models/DFNet/rt_mdnet5.pth')
    parser.add_argument("-result_path", default='./result')
    parser.add_argument("-visual_log", default=False, action='store_true')
    parser.add_argument("-visualize", default=False, action='store_true')
    parser.add_argument("-adaptive_align", default=True, action='store_false')
    parser.add_argument("-padding", default=1.2, type=float)
    parser.add_argument("-jitter", default=True, action='store_false')

    args = parser.parse_args()

    # init_seed(1234)
    ##################################################################################
    #########################Just modify opts in this script.#########################
    ######################Becuase of synchronization of options#######################
    ##################################################################################
    ## option setting
    opts['model_path'] = args.model_path
    opts['result_path'] = args.result_path
    opts['visual_log'] = args.visual_log
    opts['set_type'] = args.set_type
    opts['visualize'] = args.visualize
    opts['adaptive_align'] = args.adaptive_align
    opts['padding'] = args.padding
    opts['jitter'] = args.jitter
    ##################################################################################
    ############################Do not modify opts anymore.###########################
    ######################Becuase of synchronization of options#######################
    ##################################################################################
    print(opts)

    ## path initialization
    dataset_path = './dataset/'

    seq_home = dataset_path + opts['set_type']
    seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home, f))]

    iou_list = []
    fps_list = dict()
    bb_result = dict()
    result = dict()

    iou_list_nobb = []
    bb_result_nobb = dict()
    for num, seq in enumerate(seq_list):
        if seq != 'occBike':
            continue
        if num < -1:
            continue
        seq_path = seq_home + '/' + seq
        rgb_img_list, t_img_list, init_bbox, gt, result_path = genConfig(seq_path, opts['set_type'],
                                                                         opts['result_path'])

        iou_result, result_bb, fps, result_nobb = run_mdnet_rgbt(rgb_img_list, t_img_list, gt[0], gt, seq=seq,
                                                                 display=opts['visualize'])

        enable_frameNum = 0.
        for iidx in range(len(iou_result)):
            if (math.isnan(iou_result[iidx]) == False):
                enable_frameNum += 1.
            else:
                ## gt is not alowed
                iou_result[iidx] = 0.

        iou_list.append(iou_result.sum() / enable_frameNum)
        bb_result[seq] = result_bb
        fps_list[seq] = fps

        bb_result_nobb[seq] = result_nobb
        print(
            '{} {} : {} , total mIoU:{}, fps:{}'.format(num, seq, iou_result.mean(), sum(iou_list) / len(iou_list),
                                                        sum(fps_list.values()) / len(fps_list)))

        # Save result
        f = open(result_path, 'w+')
        if opts['set_type'] == 'GTOT':
            for i in range(len(result_bb)):
                res = '{} {} {} {} {} {} {} {}'.format(result_bb[i][0],
                                                       result_bb[i][1],
                                                       result_bb[i][0] + result_bb[i][2],
                                                       result_bb[i][1],
                                                       result_bb[i][0] + result_bb[i][2],
                                                       result_bb[i][1] + result_bb[i][3],
                                                       result_bb[i][0],
                                                       result_bb[i][1] + result_bb[i][3]
                                                       )
                f.write(res)
                f.write('\n')
        elif opts['set_type'] == 'RGBT234':
            for i in range(len(result_bb)):
                res = '{} {} {} {} '.format(result_bb[i][0], result_bb[i][1], result_bb[i][2], result_bb[i][3], )
                f.write(res)
                f.write('\n')
        f.close()

    result['bb_result'] = bb_result
    result['fps'] = fps_list
    result['bb_result_nobb'] = bb_result_nobb
    # np.save(opts['result_path'], result)
