import sys
import numpy as np
from PIL import Image, ImageFilter

import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
from ..utils import *

import matplotlib.patches as patches

import os
from ..sample_generator import *

import sys
# from pretrain_options import *

from ..img_cropper import *


class RGB_Gray_Dataset(data.Dataset):
    def __init__(self, img_dir, img_list, gt, receptive_field, opts):

        self.img_list = np.array([os.path.join(img_dir, img) for img in img_list])
        self.gt = gt

        self.batch_frames = opts['batch_frames']
        self.batch_pos = opts['batch_pos']
        self.batch_neg = opts['batch_neg']

        self.overlap_pos = opts['overlap_pos']
        self.overlap_neg = opts['overlap_neg']

        self.crop_size = opts['img_size']
        self.padding = opts['padding']

        self.index = np.random.permutation(len(self.img_list))
        self.pointer = 0

        image = Image.open(self.img_list[0]).convert('RGB')
        self.scene_generator = SampleGenerator('gaussian', image.size, trans_f=1.5, scale_f=1.2, valid=True)
        self.pos_generator = SampleGenerator('gaussian', image.size, 0.1, 1.2, 1.1, True)
        self.neg_generator = SampleGenerator('uniform', image.size, 1, 1.2, 1.1, True)

        self.receptive_field = receptive_field

        self.interval = opts['frame_interval']
        self.img_crop_model = imgCropper(opts['padded_img_size'])
        self.img_crop_model.eval()
        self.use_gpu = opts['use_gpu']
        if opts['use_gpu']:
            self.img_crop_model.gpuEnable()

    def __iter__(self):
        return self

    def __next__(self):

        next_pointer = min(self.pointer + self.batch_frames, len(self.img_list))
        idx = self.index[self.pointer:next_pointer]
        if len(idx) < self.batch_frames:
            self.index = np.random.permutation(len(self.img_list))
            next_pointer = self.batch_frames - len(idx)
            idx = np.concatenate((idx, self.index[:next_pointer]))
        self.pointer = next_pointer

        n_pos = self.batch_pos
        n_neg = self.batch_neg

        scenes_rgb = []
        scenes_t = []
        total_pos_rois = []
        total_neg_rois = []
        for i, (img_path, bbox) in enumerate(zip(self.img_list[idx], self.gt[idx])):
            image = Image.open(img_path).convert('RGB')

            image_gray = Image.blend(image.filter(ImageFilter.BoxBlur(2)).convert('L'),
                                     Image.effect_noise(image.size, 100),
                                     0.01)  # RGBT234:0.01 GTOT:5

            # if np.random.rand(1) < 0.4:
            #     if np.random.rand(1) < 0.5:
            #         image = Image.blend(image, Image.effect_noise(image.size, 50).convert('RGB'), 0.7)
            #     else:
            #         image_gray = Image.blend(image_gray, Image.effect_noise(image.size, 10), 0.5)


            image = np.asarray(image)
            image_gray = np.expand_dims(image_gray, 2).repeat(3, axis=2)  # np.asarray(image_gray)

            ishape = image.shape
            pos_examples = gen_samples(SampleGenerator('gaussian', (ishape[1], ishape[0]), 0.1, 1.2, 1.1, False), bbox,
                                       n_pos, overlap_range=self.overlap_pos)
            neg_examples = gen_samples(SampleGenerator('uniform', (ishape[1], ishape[0]), 1, 1.2, 1.1, False), bbox,
                                       n_neg, overlap_range=self.overlap_neg)

            # compute padded sample
            padded_x1 = (neg_examples[:, 0] - neg_examples[:, 2] * (self.padding - 1.) / 2.).min()
            padded_y1 = (neg_examples[:, 1] - neg_examples[:, 3] * (self.padding - 1.) / 2.).min()
            padded_x2 = (neg_examples[:, 0] + neg_examples[:, 2] * (self.padding + 1.) / 2.).max()
            padded_y2 = (neg_examples[:, 1] + neg_examples[:, 3] * (self.padding + 1.) / 2.).max()
            padded_scene_box = np.asarray((padded_x1, padded_y1, padded_x2 - padded_x1, padded_y2 - padded_y1))

            jitter_scale = 1.1 ** np.clip(3. * np.random.randn(1, 1), -2, 2)
            crop_img_size = (padded_scene_box[2:4] * ((self.crop_size, self.crop_size) / bbox[2:4])).astype('int64') * \
                            jitter_scale[0][0]
            cropped_image, cur_image_var = self.img_crop_model.crop_image(image,
                                                                          np.reshape(padded_scene_box, (1, 4)),
                                                                          crop_img_size)
            cropped_image_gray, cur_image_var_gray = self.img_crop_model.crop_image(image_gray,
                                                                                    np.reshape(padded_scene_box,
                                                                                               (1, 4)),
                                                                                    crop_img_size)
            cropped_image = cropped_image - 128.
            cropped_image_gray = cropped_image_gray - 128.
            if self.use_gpu:
                cropped_image = cropped_image.data.cpu()
                cur_image_var = cur_image_var.cpu()
                cropped_image_gray = cropped_image_gray.data.cpu()
                cur_image_var_gray = cur_image_var_gray.cpu()
            # cropped_image_4channel = torch.cat((cropped_image, cropped_image_gray), 1)
            scenes_rgb.append(cropped_image)
            scenes_t.append(cropped_image_gray)
            ## get current frame and heatmap

            rel_bbox = np.copy(bbox)
            rel_bbox[0:2] -= padded_scene_box[0:2]

            jittered_obj_size = jitter_scale[0][0] * float(self.crop_size)

            batch_num = np.zeros((pos_examples.shape[0], 1))
            pos_rois = np.copy(pos_examples)
            pos_rois[:, 0:2] -= np.repeat(np.reshape(padded_scene_box[0:2], (1, 2)), pos_rois.shape[0], axis=0)
            pos_rois = samples2maskroi(pos_rois, self.receptive_field, (jittered_obj_size, jittered_obj_size),
                                       bbox[2:4], self.padding)
            pos_rois = np.concatenate((batch_num, pos_rois), axis=1)

            batch_num = np.zeros((neg_examples.shape[0], 1))
            neg_rois = np.copy(neg_examples)
            neg_rois[:, 0:2] -= np.repeat(np.reshape(padded_scene_box[0:2], (1, 2)), neg_rois.shape[0], axis=0)
            neg_rois = samples2maskroi(neg_rois, self.receptive_field, (jittered_obj_size, jittered_obj_size),
                                       bbox[2:4], self.padding)
            neg_rois = np.concatenate((batch_num, neg_rois), axis=1)

            total_pos_rois.append(torch.from_numpy(np.copy(pos_rois).astype('float32')))
            total_neg_rois.append(torch.from_numpy(np.copy(neg_rois).astype('float32')))

        return scenes_rgb, scenes_t, total_pos_rois, total_neg_rois

    next = __next__

    def extract_regions(self, image, samples):
        regions = np.zeros((len(samples), self.crop_size, self.crop_size, 3), dtype='uint8')
        for i, sample in enumerate(samples):
            regions[i] = crop_image(image, sample, self.crop_size, self.padding, True)

        regions = regions.transpose(0, 3, 1, 2)
        regions = regions.astype('float32') - 128.
        return regions


class RegionExtractor:
    def __init__(self, image, samples, crop_size, padding, batch_size, shuffle=False):

        self.image = np.asarray(image)
        self.samples = samples
        self.crop_size = crop_size
        self.padding = padding
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.index = np.arange(len(samples))
        self.pointer = 0

        self.mean = self.image.mean(0).mean(0).astype('float32')

    def __iter__(self):
        return self

    def __next__(self):
        if self.pointer == len(self.samples):
            self.pointer = 0
            raise StopIteration
        else:
            next_pointer = min(self.pointer + self.batch_size, len(self.samples))
            index = self.index[self.pointer:next_pointer]
            self.pointer = next_pointer

            regions = self.extract_regions(index)
            regions = torch.from_numpy(regions)
            return regions

    next = __next__

    def extract_regions(self, index):
        regions = np.zeros((len(index), self.crop_size, self.crop_size, 3), dtype='uint8')
        for i, sample in enumerate(self.samples[index]):
            regions[i] = crop_image(self.image, sample, self.crop_size, self.padding)

        regions = regions.transpose(0, 3, 1, 2).astype('float32')
        regions = regions - 128.
        return regions
