import sys
import pickle
import random
import torch.optim as optim

sys.path.insert(0, './modules')
from modules.dataset.rgb_gray_dataset_separate import *
from modules.backbone.DFNet import *
from modules.pretrain_options import *
# from tracker_rtmdnet_OTB import *
import numpy as np

import argparse

from tensorboardX import SummaryWriter


def init_seed(seed=None):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


def set_optimizer(model, lr_base, lr_mult=pretrain_opts['lr_mult'], momentum=pretrain_opts['momentum'],
                  w_decay=pretrain_opts['w_decay']):
    params = model.get_learnable_params()
    param_list = []
    for k, p in params.items():
        lr = lr_base
        for l, m in lr_mult.items():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr': lr})
    optimizer = optim.SGD(param_list, lr=lr, momentum=momentum, weight_decay=w_decay)
    return optimizer


def train_mdnet():
    ## set image directory
    if pretrain_opts['set_type'] == 'OTB':
        img_home = './dataset/tracking/OTB/'
        data_path = './otb-vot15.pkl'
    if pretrain_opts['set_type'] == 'VOT':
        img_home = './dataset/tracking/VOT/'
        data_path = './vot-otb.pkl'
    if pretrain_opts['set_type'] == 'IMAGENET':
        img_home = './dataset/ILSVRC/Data/VID/train/'
        data_path = './data/imagenet_refine.pkl'

    # create summary writer tensorboardX
    if not os.path.exists(pretrain_opts['log_dir']):
        os.makedirs(pretrain_opts['log_dir'])

    summary_writer = SummaryWriter(pretrain_opts['log_dir'])

    ## Init dataset ##
    with open(data_path, 'rb') as fp:
        data = pickle.load(fp)

    train_video_numbers = len(data)

    ## Init model ##
    model = DFNet(pretrain_opts['init_model_path'], train_video_numbers)
    if pretrain_opts['adaptive_align']:
        align_h = model.roi_pool_model.pooled_height
        align_w = model.roi_pool_model.pooled_width
        spatial_s = model.roi_pool_model.spatial_scale
        model.roi_pool_model = PrRoIPool2D(align_h, align_w, spatial_s)

    if pretrain_opts['use_gpu']:
        model = model.cuda()
    model.set_learnable_params(pretrain_opts['ft_layers'])
    model.train()

    dataset = [None] * train_video_numbers
    for real_video_index, (seqname, seq) in enumerate(data.items()):
        img_list = seq['images']
        gt = seq['gt']
        if pretrain_opts['set_type'] == 'OTB':
            img_dir = os.path.join(img_home, seqname + '/img')
        if pretrain_opts['set_type'] == 'VOT':
            img_dir = img_home + seqname
        if pretrain_opts['set_type'] == 'IMAGENET':
            img_dir = img_home + seqname
        dataset[real_video_index] = RGB_Gray_Dataset(img_dir, img_list, gt, model.receptive_field, pretrain_opts)

    ## Init criterion and optimizer ##
    binaryCriterion = BinaryLoss()
    interDomainCriterion = nn.CrossEntropyLoss()
    evaluator = Precision()
    optimizer = set_optimizer(model, pretrain_opts['lr'])

    best_score = 0.
    batch_cur_idx = 0
    for epoch in range(pretrain_opts['n_cycles']):
        print("==== Start Cycle %d ====" % epoch)
        k_list = np.random.permutation(train_video_numbers)
        train_prec = np.zeros(train_video_numbers)
        train_totalClassLoss = np.zeros(train_video_numbers)
        train_totalInterClassLoss = np.zeros(train_video_numbers)
        for step, real_video_index in enumerate(k_list):
            tic = time.time()
            try:
                cropped_scenes_rgb, cropped_scenes_t, pos_rois, neg_rois = dataset[real_video_index].next()
            except:
                continue

            try:
                for sidx in range(0, len(cropped_scenes_rgb)):
                    cur_scene_rgb = cropped_scenes_rgb[sidx]
                    cur_scene_t = cropped_scenes_t[sidx]
                    cur_pos_rois = pos_rois[sidx]
                    cur_neg_rois = neg_rois[sidx]

                    cur_scene_rgb = Variable(cur_scene_rgb)
                    cur_scene_t = Variable(cur_scene_t)
                    cur_pos_rois = Variable(cur_pos_rois)
                    cur_neg_rois = Variable(cur_neg_rois)
                    if pretrain_opts['use_gpu']:
                        cur_scene_rgb = cur_scene_rgb.cuda()
                        cur_scene_t = cur_scene_t.cuda()
                        cur_pos_rois = cur_pos_rois.cuda()
                        cur_neg_rois = cur_neg_rois.cuda()
                    cur_feat_map = model(R=cur_scene_rgb, T=cur_scene_t, k=real_video_index, out_layer='conv3')

                    cur_pos_feats = model.roi_pool_model(cur_feat_map, cur_pos_rois)
                    cur_pos_feats = cur_pos_feats.view(cur_pos_feats.size(0), -1)
                    cur_neg_feats = model.roi_pool_model(cur_feat_map, cur_neg_rois)
                    cur_neg_feats = cur_neg_feats.view(cur_neg_feats.size(0), -1)

                    if sidx == 0:
                        pos_feats = [cur_pos_feats]
                        neg_feats = [cur_neg_feats]
                    else:
                        pos_feats.append(cur_pos_feats)
                        neg_feats.append(cur_neg_feats)
                feat_dim = cur_neg_feats.size(1)
                pos_feats = torch.stack(pos_feats, dim=0).view(-1, feat_dim)
                neg_feats = torch.stack(neg_feats, dim=0).view(-1, feat_dim)
            except:
                continue

            pos_score = model(feat=pos_feats, k=real_video_index, in_layer='fc4')
            neg_score = model(feat=neg_feats, k=real_video_index, in_layer='fc4')

            class_loss = binaryCriterion(pos_score, neg_score)
            train_totalClassLoss[real_video_index] = class_loss.item()

            ## inter frame classification

            interclass_label = Variable(torch.zeros((pos_score.size(0))).long())
            if pretrain_opts['use_gpu']:
                interclass_label = interclass_label.cuda()
            total_interclass_score = pos_score[:, 1].contiguous()
            total_interclass_score = total_interclass_score.view((pos_score.size(0), 1))

            K_perm = np.random.permutation(train_video_numbers)
            K_perm = K_perm[0:100]
            for cidx in K_perm:
                if real_video_index == cidx:
                    continue
                else:
                    interclass_score = model(feat=pos_feats, k=cidx, in_layer='fc4')
                    total_interclass_score = torch.cat((total_interclass_score,
                                                        interclass_score[:, 1].contiguous().view(
                                                            (interclass_score.size(0), 1))), dim=1)

            interclass_loss = interDomainCriterion(total_interclass_score, interclass_label)
            train_totalInterClassLoss[real_video_index] = interclass_loss.item()

            total_loss = class_loss + 0.1 * interclass_loss
            total_loss.backward()
            totalstep = epoch * train_video_numbers + step
            summary_writer.add_scalar('train/BinLoss', class_loss.data, totalstep)
            summary_writer.add_scalar('train/interLoss', interclass_loss.data, totalstep)
            summary_writer.add_scalar('train/TotalLoss', total_loss.data, totalstep)

            batch_cur_idx += 1
            if (batch_cur_idx % pretrain_opts['seqbatch_size']) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), pretrain_opts['grad_clip'])
                optimizer.step()
                model.zero_grad()
                batch_cur_idx = 0

            ## evaulator
            train_prec[real_video_index] = evaluator(pos_score, neg_score)
            ## computation latency
            toc = time.time() - tic

            print("Epoch %2d, step %2d (real_video_index %2d), BinLoss %.3f, Prec %.3f, interLoss %.3f, Time %.3f" %
                  (epoch, step, real_video_index, class_loss.item(), train_prec[real_video_index],
                   train_totalInterClassLoss[real_video_index], toc))

        cur_score = train_prec.mean()
        print("Mean Precision: %.3f Class Loss: %.3f Inter Loss: %.3f" % (
            train_prec.mean(), train_totalClassLoss.mean(), train_totalInterClassLoss.mean()))
        summary_writer.add_scalar('train/Mean_Precision', train_prec.mean(), epoch)
        summary_writer.add_scalar('train/Class_Loss', train_totalClassLoss.mean(), epoch)
        summary_writer.add_scalar('train/Inter_Loss', train_totalInterClassLoss.mean(), epoch)

        best_score = 0  # TODO best_score=0 to Save every epoch

        if cur_score > best_score:
            best_score = cur_score
        if pretrain_opts['use_gpu']:
            model = model.cpu()
        states = {'shared_layers': model.layers.state_dict(), 'RGB_layers': model.RGB_layers.state_dict(),
                  'T_layers': model.T_layers.state_dict(), 'conv1weight': model.conv1weight,
                  'conv1bias': model.conv1bias.data, 'conv2weight': model.conv2weight.data,
                  'conv2bias': model.conv2bias.data, 'conv3weight': model.conv3weight.data,
                  'conv3bias': model.conv3bias.data}
        print("Save model to %s" % pretrain_opts['model_path'])
        torch.save(states, pretrain_opts['model_path'].replace('.pth', str(epoch) + '.pth'))
        if pretrain_opts['use_gpu']:
            model = model.cuda()


if __name__ == "__main__":
    init_seed(1234)
    parser = argparse.ArgumentParser()
    parser.add_argument("-set_type", default='IMAGENET')
    parser.add_argument("-padding_ratio", default=5., type=float)
    parser.add_argument("-model_path", default="./models/rt_mdnet.pth", help="model path")
    parser.add_argument("-frame_interval", default=1, type=int,
                        help="frame interval in batch. ex) interval=1 -> [1 2 3 4 5], interval=2 ->[1 3 5]")
    parser.add_argument("-init_model_path", default="./models/imagenet-vgg-m.mat")
    parser.add_argument("-batch_frames", default=8, type=int)
    parser.add_argument("-lr", default=0.0001, type=float)
    parser.add_argument("-batch_pos", default=64, type=int)
    parser.add_argument("-batch_neg", default=196, type=int)
    parser.add_argument("-n_cycles", default=1000, type=int)
    parser.add_argument("-adaptive_align", default=True, action='store_false')
    parser.add_argument("-seqbatch_size", default=50, type=int)

    args = parser.parse_args()

    ##################################################################################
    #########################Just modify opts in this script.#########################
    ######################Becuase of synchronization of options#######################
    ##################################################################################
    ##option setting
    pretrain_opts['set_type'] = args.set_type
    pretrain_opts['padding_ratio'] = args.padding_ratio
    pretrain_opts['padded_img_size'] = pretrain_opts['img_size'] * int(pretrain_opts['padding_ratio'])
    pretrain_opts['model_path'] = args.model_path
    pretrain_opts['frame_interval'] = args.frame_interval
    pretrain_opts['init_model_path'] = args.init_model_path
    pretrain_opts['batch_frames'] = args.batch_frames
    pretrain_opts['lr'] = args.lr
    pretrain_opts['batch_pos'] = args.batch_pos  # original = 64
    pretrain_opts['batch_neg'] = args.batch_neg  # original = 192
    pretrain_opts['n_cycles'] = args.n_cycles
    pretrain_opts['adaptive_align'] = args.adaptive_align
    pretrain_opts['seqbatch_size'] = args.seqbatch_size
    ##################################################################################
    ############################Do not modify opts anymore.###########################
    ######################Becuase of synchronization of options#######################
    ##################################################################################

    print(pretrain_opts)
    train_mdnet()
