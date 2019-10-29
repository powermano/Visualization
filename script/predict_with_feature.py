# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import cv2
import numpy as np

np.set_printoptions(threshold=np.inf)
import mxnet as mx
import random
import time
import copy
import math
import argparse
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

root_dir = os.path.abspath(__file__).split('script')[0]
sys.path.insert(0, root_dir + '/script')

from utils.pyramid_resizer.utils import img_pyramid_resizer
from utils.pyramid_resizer.resizer import Resizer
from collections import namedtuple

Batch = namedtuple('Batch', ['data'])


class AntiSpoofingPredictor():
    def __init__(self, args, process_method):
        self.config = args
        self.process_method = process_method
        # evaluate
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

        self.prob_list = []
        self.label_list = []

    def parse_file_dir(self):
        img_info_list = []
        if self.config.file is not None:
            with open(self.config.file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                try:
                    image_info = dict()
                    data = line.strip().split(' ')
                    if len(data) < 2:  # at least has img_path, rects(4), liveness(1)
                        continue
                    img_path = ' '.join(data[:-5])
                    img_path = img_path.replace('\\ ', '\ ')
                    img_path = img_path.replace('\ ', ' ')
                    if not os.path.exists(img_path):
                        logging.warn('image `{}` is not exist.'.format(img_path))
                        continue
                    image_info['img_path'] = img_path
                    image_info['img_name'] = ' '.join(data[:-5]).split('/')[-1]
                    rects = [float(x) for x in data[-5:-1]]
                    image_info['rects'] = rects
                    image_info['liveness'] = np.round(float(data[-1]))

                except Exception as e:
                    continue
                img_info_list.append(image_info)

        if self.config.dir is not None:
            img_walk_list = os.walk(self.config.dir)
            for one in img_walk_list:
                for two in one[-1]:
                    image_info = dict()
                    img_path = os.path.join(one[0], two)
                    image_info['img_path'] = img_path
                    image_info['img_name'] = img_path.split('/')[-1]
                    img_h, img_w, _ = cv2.imread(img_path).shape
                    image_info['rects'] = [0, 0, img_w - 1, img_h - 1]
                    if 'live' in img_path:
                        image_info['liveness'] = 1.0
                    elif 'spoof' in img_path:
                        image_info['liveness'] = 0.0
                    else:
                        continue

                    img_info_list.append(image_info)

        return img_info_list

    def draw_feature(self, img, feat, name, select_index):
        color = [0, 0, 5]
        if not os.path.exists('./{}_feat'.format(name)):
            os.makedirs('./{}_feat'.format(name))
        for idx in range(feat.shape[-1]):
            img_tmp = np.zeros((img.shape[0], img.shape[1], 3))
            fea_tmp = np.zeros((img.shape[0], img.shape[1], 3))
            for i in range(3):
                img_tmp[:, :, i] = img[:, :, 0]
                fea_tmp[:, :, i] = cv2.resize(feat[select_index, :, :, idx], (img.shape[1], img.shape[0])) * color[i]
            save_img = img_tmp * 0.5 + fea_tmp * 0.5
            cv2.imwrite('./{}_feat/{}_channel_{}.png'.format(name, name, idx), save_img)

    def post_process_batch(self, outputs, imgs):
        if outputs[1].shape[-1] == 1:
            # probs = mx.nd.sigmoid(outputs[1], axis=1).asnumpy()
            probs = mx.nd.reshape(outputs[1], (-1, 2))
            probs = mx.nd.softmax(outputs[1], axis=1)[:, 1].asnumpy()
            feature_output = (mx.nd.sum(mx.nd.transpose(outputs[0], (0, 2, 3, 1)), axis=(1, 2)) / 4).asnumpy()
        elif outputs[1].shape[-1] == 2:
            # probs = mx.nd.sigmoid(outputs[0][:, [1]], axis=1).asnumpy()
            probs = mx.nd.softmax(outputs[1], axis=1)[:, 1].asnumpy()
            feature_output = outputs[0].asnumpy()
        else:
            assert False, 'not support network outputs shape: {}'.format(outputs[0].shape)
        preds = probs.copy()
        preds[preds > self.config.thresh] = 1.0
        preds[preds <= self.config.thresh] = 0.0

        if len(outputs) == 5:
            select_index = 0
            img = imgs[select_index] * 128. + 128.

            res2_feat = outputs[1].asnumpy().astype(np.float32)
            res3_feat = outputs[2].asnumpy().astype(np.float32)
            res4_feat = outputs[3].asnumpy().astype(np.float32)
            res5_feat = outputs[4].asnumpy().astype(np.float32)

            self.draw_feature(img, res2_feat, 'res2', select_index)
            self.draw_feature(img, res3_feat, 'res3', select_index)
            self.draw_feature(img, res4_feat, 'res4', select_index)
            self.draw_feature(img, res5_feat, 'res5', select_index)

        return preds, probs, feature_output

    def get_save_string(self, img_info, pred, prob, feature):
        '''
        return:
            save_string: `img_path(1) rects(4) liveness_label(1) liveness_pred(1)\n`
        '''
        save_string = ''
        # img path
        save_string += img_info['img_path']
        save_string += ' '
        # rects
        save_string += ' '.join([str(rect) for rect in img_info['rects']])
        # liveness
        save_string += ' {}'.format(float(img_info['liveness']))
        save_string += ' {}'.format(float(pred))
        save_string += ' {}'.format(float(prob))
        # set feature
        if self.config.save-feature:
            save_string += ' '
            save_string += ' '.join([str(x) for x in feature.tolist()])

        save_string += '\n'

        return save_string

    def save_bad_case(self, img_info, save_prefix):
        img_name_prefix = img_info['img_path'].split('/')[-2] + '_' + \
                          '.'.join(img_info['img_name'].split('.')[:-1]) + '_' \
                          + time.strftime("%Y%m%d%H%M%S", time.localtime())
        rects = img_info['rects']

        img_full = cv2.imdecode(np.fromfile(img_info['img_path'], dtype=np.uint8), -1)
        cv2.rectangle(img_full, (int(rects[0]), int(rects[1])), (int(rects[2]), int(rects[3])), (0, 255, 255), 2)
        img_crop, label = self.process_method(img_info, self.config.expand_ratio)
        cv2.imwrite(os.path.join(save_prefix, img_name_prefix + '_full.jpg'), img_full)
        cv2.imwrite(os.path.join(save_prefix, img_name_prefix + '_crop.jpg'), img_crop)

    def predict(self):
        # parse self.config
        img_info_list = self.parse_file_dir()
        ctx = [mx.cpu()] if self.config.gpus == '-1' else [mx.gpu(int(i)) for i in self.config.gpus.split(',')]
        input_shape = [int(x) for x in self.config.input_shape.split(',')]

        logging.info('Loading model...')
        sym, arg_params, aux_params = mx.model.load_checkpoint(self.config.prefix, self.config.epoch)
        mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        if self.config.is_qnn:
            mod.bind(for_training=False,
                     data_shapes=[('data', (self.config.batch_size, input_shape[0], input_shape[1], input_shape[2]))],
                     label_shapes=mod._label_shapes)
        else:
            mod.bind(for_training=False,
                     data_shapes=[('data', (self.config.batch_size, input_shape[2], input_shape[0], input_shape[1]))],
                     label_shapes=mod._label_shapes)
        mod.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)
        logging.info('Load model success.')

        if self.config.save is not None:
            if os.path.exists(self.config.save):
                with open(self.config.save, 'r') as f:
                    has_saved_result_lines = f.readlines()
                has_saved_idx = np.maximum(int(len(has_saved_result_lines) / self.config.batch_size) - 1, 0)
                save_result = open(self.config.save, 'w')
                for line in has_saved_result_lines[:has_saved_idx * self.config.batch_size]:
                    save_result.write(line)
            else:
                save_result = open(self.config.save, 'w')
                if self.config.save-badcase:
                    save_bad_result = open('./badcase_outputs.txt', 'w')
                has_saved_idx = 0

        num_img = len(img_info_list)
        num_iter = int(num_img / self.config.batch_size)
        if num_img % self.config.batch_size != 0:
            num_iter += 1

        # init idx
        from_idx = has_saved_idx * self.config.batch_size
        to_idx = from_idx + self.config.batch_size
        for idx in range(has_saved_idx, num_iter):
            logging.info('{} / {} ...'.format(idx + 1, num_iter))
            from_idx = idx * self.config.batch_size
            to_idx = from_idx + self.config.batch_size
            if to_idx >= num_img:
                batch_img_info = img_info_list[from_idx:to_idx]
                batch_img_info += img_info_list[0:(to_idx - num_img)]
            else:
                batch_img_info = img_info_list[from_idx:to_idx]

            # generate a batch data.
            imgs = []
            labels = []
            for img_info in batch_img_info:
                img, label = self.process_method(img_info, self.config.expand_ratio)
                if self.config.data_type in [1, 5, 6]:
                    assert self.config.input_format == 'gray'
                    assert input_shape[2] == 1, 'gray image must has 1 channel.'
                elif self.config.data_type in [7]:
                    assert self.config.input_format == 'yuv'
                    assert input_shape[2] == 3
                else:
                    if self.config.input_format == 'rgb':
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    elif self.config.input_format == 'bgr':
                        img = img
                    elif self.config.input_format == 'gray':
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    elif self.config.input_format == 'yuv':
                        ### opencv
                        # img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                        ### same to hardware
                        img = bgr2yuv444(img)
                        ### GluonHorizon
                        # trans = mx.image.CenteredYUVAug(to_rgb=False)
                        # img = mx.nd.array(img)
                        # img = trans(img)
                        # img = img.asnumpy()
                    else:
                        assert False, 'not support input format: {}.'.format(self.config.input_format)
                img = np.reshape(img, input_shape)
                # normalize image
                img = (img - 128.0) / 128.0

                if not self.config.is_qnn:
                    img = np.transpose(img, (2, 0, 1))
                imgs.append(img)
                labels.append(label)
            imgs = np.array(imgs)
            labels = np.array(labels)

            ### forward
            mod.forward(Batch([mx.nd.array(imgs)]))
            outputs = mod.get_outputs()

            preds, probs, feature = self.post_process_batch(outputs, imgs)

            # save result file
            if self.config.save is not None:
                if to_idx >= num_img:
                    num_save_idx = self.config.batch_size - (to_idx - num_img)
                else:
                    num_save_idx = self.config.batch_size
                for save_idx in range(num_save_idx):
                    img_info = batch_img_info[save_idx]
                    this_pred = preds[save_idx]
                    this_prob = probs[save_idx]
                    this_feature = feature[save_idx, :]

                    # evaluate results
                    this_label = img_info['liveness']
                    self.tp += 1 if (this_label == 1.0 and this_pred == 1.0) else 0
                    self.tn += 1 if (this_label != 1.0 and this_pred != 1.0) else 0
                    self.fp += 1 if (this_label != 1.0 and this_pred == 1.0) else 0
                    self.fn += 1 if (this_label == 1.0 and this_pred != 1.0) else 0

                    if self.config.save-badcase:
                        if this_label == 1.0 and this_pred != 1.0:
                            save_bad_string = self.get_save_string(img_info, this_pred, this_prob, this_feature)
                            save_bad_result.write(save_bad_string)
                        if (this_label != 1.0 and this_pred == 1.0):
                            save_bad_string = self.get_save_string(img_info, this_pred, this_prob, this_feature)
                            save_bad_result.write(save_bad_string)

                    # save bad case image
                    # if (this_label == 1.0 and this_pred != 1.0):
                    #     if not os.path.exists('./live_bad_case'):
                    #         os.makedirs('./live_bad_case')
                    #     self.save_bad_case(img_info, './live_bad_case')
                    # if (this_label != 1.0 and this_pred == 1.0):
                    #     if not os.path.exists('./spoof_bad_case'):
                    #         os.makedirs('./spoof_bad_case')
                    #     self.save_bad_case(img_info, './spoof_bad_case')

                    # save results
                    save_string = self.get_save_string(img_info, this_pred, this_prob, this_feature)
                    save_result.write(save_string)

                    # save prob/label_list
                    self.prob_list.append(this_prob)
                    self.label_list.append(this_label)

    def evaluate(self):
        self.TAR = float(self.tp) / float(self.tp + self.fn + 1e-9)  # TAR or LAR(live_accept_rate)
        self.TRR = float(self.tn) / float(self.tn + self.fp + 1e-9)  # TRR or SRR(spoof_reject_rate)
        self.precision = float(self.tp + self.tn) / float(self.tp + self.tn + self.fp + self.fn + 1e-9)  # Precision

        logging.info('### Results ###')
        logging.info('The result of model `{}` with epoch={} is:'.format(self.config.prefix, self.config.epoch))
        logging.info('Thresh:    {:.2f}'.format(self.config.thresh))
        logging.info('TAR:       {:.2f}%'.format(self.TAR * 100))
        logging.info('TRR:       {:.2f}%'.format(self.TRR * 100))
        logging.info('precision: {:.2f}%'.format(self.precision * 100))

    def evaluate_multi_thresh(self):
        prob_array = np.array(self.prob_list)
        label_array = np.array(self.label_list)
        thresh_list = np.arange(0.05, 1, 0.05)

        logging.info('### Results with different thresh ###')
        logging.info('Thresh\tTAR\tTRR\tprecision')
        for thresh in thresh_list:
            pred_array = copy.deepcopy(prob_array)
            pred_array[pred_array > thresh] = 1.0
            pred_array[pred_array <= thresh] = 0.0

            tp = 0.0
            tn = 0.0
            fp = 0.0
            fn = 0.0

            for i in range(len(self.prob_list)):
                this_pred = pred_array[i]
                this_label = label_array[i]
                tp += 1 if (this_label == 1.0 and this_pred == 1.0) else 0
                tn += 1 if (this_label != 1.0 and this_pred != 1.0) else 0
                fp += 1 if (this_label != 1.0 and this_pred == 1.0) else 0
                fn += 1 if (this_label == 1.0 and this_pred != 1.0) else 0

            TAR = float(tp) / float(tp + fn + 1e-9)  # TAR or LAR(live_accept_rate)
            TRR = float(tn) / float(tn + fp + 1e-9)  # TRR or SRR(spoof_reject_rate)
            precision = float(tp + tn) / float(tp + tn + fp + fn + 1e-9)  # Precision

            logging.info('{:.2f}\t{:.2f}%\t{:.2f}%\t{:.2f}%'.format(thresh, TAR * 100, TRR * 100, precision * 100))


def bgr2yuv444(img_bgr):
    img_h = img_bgr.shape[0]
    img_w = img_bgr.shape[1]
    uv_start_idx = img_h * img_w
    v_size = int(img_h * img_w / 4)

    def _trans(img_uv):
        img_uv = img_uv.reshape(int(math.ceil(img_h / 2.0)), int(math.ceil(img_w / 2.0)), 1)
        img_uv = np.repeat(img_uv, 2, axis=0)
        img_uv = np.repeat(img_uv, 2, axis=1)
        return img_uv

    img_yuv420sp = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV_I420)
    img_yuv420sp = img_yuv420sp.flatten()
    img_y = img_yuv420sp[:uv_start_idx].reshape((img_h, img_w, 1))
    img_u = img_yuv420sp[uv_start_idx:uv_start_idx + v_size]
    img_v = img_yuv420sp[uv_start_idx + v_size:uv_start_idx + 2 * v_size]
    img_u = _trans(img_u)
    img_v = _trans(img_v)
    img_yuv444 = np.concatenate((img_y, img_u, img_v), axis=2)
    return img_yuv444


def _read_yuv(path, dims):
    with open(path, 'rb') as f:
        data = f.read(np.prod(dims))
        img_y = np.frombuffer(data, dtype=np.uint8).reshape(dims[0], dims[1])

    return img_y


def process_method_1(data_dict, expand_ratio=1.0):
    if data_dict['img_path'].split('.')[-1] != 'yuv':
        img = cv2.imdecode(np.fromfile(data_dict['img_path'], dtype=np.uint8), 1)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = _read_yuv(data_dict['img_path'], (1080, 1920))
        img_gray = img
    if img is None:
        assert False, 'image `{}` is empty.'.format(data_dict['img_path'])

    img_gray = np.clip(img_gray, 16, 235)
    img_h, img_w = img_gray.shape[:2]
    rects = copy.deepcopy(data_dict['rects'])
    liveness = data_dict['liveness']

    img_big = np.zeros((1080, 1920))
    try:
        img_big[:img_h, :img_w] = img_gray
    except:
        assert False, 'jumping because paste fail: {}'.format(data_dict['img_path'])

    img_big = img_big.astype(np.uint8)
    bbox = [int(x) for x in rects]

    cropped_imgs, bbox_lists, layer = img_pyramid_resizer(img=img_big, rois=[bbox], norm_len=int(128.0 / expand_ratio),
                                                          input_hw=[128, 128],
                                                          use_pyramid=True, use_resizer=True, im_w=img_w, im_h=img_h,
                                                          restrict_bound=True, norm_method='none')

    image = cropped_imgs[0]
    image = np.clip(image, 16, 235)

    return image, liveness


def process_method_2(data_dict, expand_ratio=0.0):
    if data_dict['img_path'].split('.')[-1] != 'yuv':
        img = cv2.imdecode(np.fromfile(data_dict['img_path'], dtype=np.uint8), 1)
    else:
        img_gray = _read_yuv(data_dict['img_path'], (1080, 1920))
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        img[:, :, 0] = img_gray
        img[:, :, 1] = img_gray
        img[:, :, 2] = img_gray
    if img is None:
        assert False, 'image `{}` is empty.'.format(data_dict['img_path'])

    rects = copy.deepcopy(data_dict['rects'])
    liveness = data_dict['liveness']

    img_h, img_w = img.shape[:2]

    if img_h < 128 or img_w < 128:
        assert False, 'jumping because image too small: {}'.format(data_dict['img_path'])

    center_x = int((rects[0] + rects[2]) / 2)
    center_y = int((rects[1] + rects[3]) / 2)
    bbox = [center_x - 64, center_y - 64, center_x + 64, center_y + 64]
    if center_x - 64 < 0:
        bbox[0] = 0
        bbox[2] = 128
    if center_y - 64 < 0:
        bbox[1] = 0
        bbox[3] = 128
    if center_x + 64 > img.shape[1]:
        bbox[0] = img.shape[1] - 128
        bbox[2] = img.shape[1]
    if center_y + 64 > img.shape[0]:
        bbox[1] = img.shape[0] - 128
        bbox[3] = img.shape[0]

    img_final = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    if img_final.shape[:2] != (128, 128):
        assert False, 'shape wrong, except {}, but got {}'.format('(128, 128)', str(img_final.shape))

    return img_final, liveness


def process_method_3(data_dict, expand_ratio=1.0):
    if data_dict['img_path'].split('.')[-1] != 'yuv':
        img = cv2.imdecode(np.fromfile(data_dict['img_path'], dtype=np.uint8), 1)
    else:
        img_gray = _read_yuv(data_dict['img_path'], (1080, 1920))
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        img[:, :, 0] = img_gray
        img[:, :, 1] = img_gray
        img[:, :, 2] = img_gray
    if img is None:
        assert False, 'image `{}` is empty.'.format(data_dict['img_path'])

    rects = copy.deepcopy(data_dict['rects'])
    liveness = data_dict['liveness']

    img_h, img_w = img.shape[:2]

    w = rects[2] - rects[0]
    h = rects[3] - rects[1]
    expand_ratio = expand_ratio
    bbox1 = [np.maximum(rects[0] - w * (expand_ratio - 1) / 2., 0),
             np.maximum(rects[1] - h * (expand_ratio - 1) / 2., 0),
             np.minimum(rects[2] + w * (expand_ratio - 1) / 2., img_w),
             np.minimum(rects[3] + h * (expand_ratio - 1) / 2., img_h)]

    # rec_w = bbox1[2] - bbox1[0]
    # rec_h = bbox1[3] - bbox1[1]
    # rec_expand_ratio = 1.125
    # bbox2 = [np.maximum(bbox1[0] - rec_w * (rec_expand_ratio - 1) / 2., 0),
    #         np.maximum(bbox1[1] - rec_h * (rec_expand_ratio - 1) / 2., 0),
    #         np.minimum(bbox1[2] + rec_w * (rec_expand_ratio - 1) / 2., img_w),
    #         np.minimum(bbox1[3] + rec_h * (rec_expand_ratio - 1) / 2., img_h)]
    bbox2 = bbox1

    bbox2 = [int(x) for x in bbox2]
    img_final = img[bbox2[1]:bbox2[3], bbox2[0]:bbox2[2]]
    img_final = cv2.resize(img_final, (128, 128))

    return img_final, liveness


def process_method_4(data_dict, expand_ratio=1.0):
    if data_dict['img_path'].split('.')[-1] != 'yuv':
        img = cv2.imdecode(np.fromfile(data_dict['img_path'], dtype=np.uint8), 1)
    else:
        img_gray = _read_yuv(data_dict['img_path'], (1080, 1920))
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        img[:, :, 0] = img_gray
        img[:, :, 1] = img_gray
        img[:, :, 2] = img_gray
    if img is None:
        assert False, 'image `{}` is empty.'.format(data_dict['img_path'])

    rects = copy.deepcopy(data_dict['rects'])
    liveness = data_dict['liveness']

    img_h, img_w = img.shape[:2]

    w = rects[2] - rects[0]
    h = rects[3] - rects[1]
    if h > w:
        origin = rects[0] + rects[2]
        rects[0] = np.maximum((origin / 2. - h / 2.), 0)
        rects[2] = np.minimum((origin / 2. + h / 2.), img_w)
    else:
        origin = rects[1] + rects[3]
        rects[1] = np.maximum((origin / 2. - w / 2.), 0)
        rects[3] = np.minimum((origin / 2. + w / 2.), img_h)

    w = rects[2] - rects[0]
    h = rects[3] - rects[1]
    expand_ratio = expand_ratio
    bbox1 = [np.maximum(rects[0] - w * (expand_ratio - 1) / 2., 0),
             np.maximum(rects[1] - h * (expand_ratio - 1) / 2., 0),
             np.minimum(rects[2] + w * (expand_ratio - 1) / 2., img_w),
             np.minimum(rects[3] + h * (expand_ratio - 1) / 2., img_h)]

    # rec_w = bbox1[2] - bbox1[0]
    # rec_h = bbox1[3] - bbox1[1]
    # rec_expand_ratio = 1.125
    # bbox2 = [np.maximum(bbox1[0] - rec_w * (rec_expand_ratio - 1) / 2., 0),
    #         np.maximum(bbox1[1] - rec_h * (rec_expand_ratio - 1) / 2., 0),
    #         np.minimum(bbox1[2] + rec_w * (rec_expand_ratio - 1) / 2., img_w),
    #         np.minimum(bbox1[3] + rec_h * (rec_expand_ratio - 1) / 2., img_h)]
    bbox2 = bbox1

    bbox2 = [int(x) for x in bbox2]
    img_final = img[bbox2[1]:bbox2[3], bbox2[0]:bbox2[2]]
    img_final = cv2.resize(img_final, (128, 128))

    return img_final, liveness


def process_method_5(data_dict, expand_ratio=1.0):
    if data_dict['img_path'].split('.')[-1] != 'yuv':
        img = cv2.imdecode(np.fromfile(data_dict['img_path'], dtype=np.uint8), 1)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = _read_yuv(data_dict['img_path'], (1080, 1920))
        img_gray = img
    if img is None:
        assert False, 'image `{}` is empty.'.format(data_dict['img_path'])

    img_gray = np.clip(img_gray, 16, 235)
    img_h, img_w = img_gray.shape[:2]
    rects = copy.deepcopy(data_dict['rects'])
    liveness = data_dict['liveness']

    img_big = np.zeros((1080, 1920))
    try:
        img_big[:img_h, :img_w] = img_gray
    except:
        assert False, 'jumping because paste fail: {}'.format(data_dict['img_path'])

    img_big = img_big.astype(np.uint8)
    bbox = [int(x) for x in rects]

    cropped_imgs, bbox_lists, layer = img_pyramid_resizer(img=img_big, rois=[bbox], norm_len=int(128.0 / expand_ratio),
                                                          input_hw=[128, 128],
                                                          use_pyramid=True, use_resizer=True, im_w=img_w, im_h=img_h,
                                                          restrict_bound=True, norm_method='sqrt_area')

    image = cropped_imgs[0]
    image = np.clip(image, 16, 235)

    return image, liveness


resizer = Resizer(target='x1', img_type='y_only', dst_h=128, dst_w=128)


def process_method_6(data_dict, expand_ratio=1.0):
    if data_dict['img_path'].split('.')[-1] != 'yuv':
        img = cv2.imdecode(np.fromfile(data_dict['img_path'], dtype=np.uint8), 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = _read_yuv(data_dict['img_path'], (1080, 1920))
    if img is None:
        assert False, 'image `{}` is empty.'.format(data_dict['img_path'])

    rects = copy.deepcopy(data_dict['rects'])
    liveness = data_dict['liveness']

    img_h, img_w = img.shape[:2]

    w = rects[2] - rects[0]
    h = rects[3] - rects[1]
    if h > w:
        origin = rects[0] + rects[2]
        rects[0] = np.maximum((origin / 2. - h / 2.), 0)
        rects[2] = np.minimum((origin / 2. + h / 2.), img_w)
    else:
        origin = rects[1] + rects[3]
        rects[1] = np.maximum((origin / 2. - w / 2.), 0)
        rects[3] = np.minimum((origin / 2. + w / 2.), img_h)

    w = rects[2] - rects[0]
    h = rects[3] - rects[1]
    expand_ratio = expand_ratio
    bbox1 = [np.maximum(rects[0] - w * (expand_ratio - 1) / 2., 0),
             np.maximum(rects[1] - h * (expand_ratio - 1) / 2., 0),
             np.minimum(rects[2] + w * (expand_ratio - 1) / 2., img_w),
             np.minimum(rects[3] + h * (expand_ratio - 1) / 2., img_h)]

    # rec_w = bbox1[2] - bbox1[0]
    # rec_h = bbox1[3] - bbox1[1]
    # rec_expand_ratio = 1.125
    # bbox2 = [np.maximum(bbox1[0] - rec_w * (rec_expand_ratio - 1) / 2., 0),
    #         np.maximum(bbox1[1] - rec_h * (rec_expand_ratio - 1) / 2., 0),
    #         np.minimum(bbox1[2] + rec_w * (rec_expand_ratio - 1) / 2., img_w),
    #         np.minimum(bbox1[3] + rec_h * (rec_expand_ratio - 1) / 2., img_h)]
    bbox2 = bbox1

    bbox2 = [int(x) for x in bbox2]
    img_final = img[bbox2[1]:bbox2[3], bbox2[0]:bbox2[2]]
    img_final = resizer.run([img_final], [[0, 0, img_final.shape[1] - 1, img_final.shape[0] - 1]])[0]

    return img_final, liveness


def process_method_7(data_dict, expand_ratio=1.0):
    ### read rgb
    img = cv2.imdecode(np.fromfile(data_dict['img_path'], dtype=np.uint8), -1)
    if img is None:
        assert False, 'image `{}` is empty.'.format(data_dict['img_path'])

    img_h, img_w = img.shape[:2]
    rects = copy.deepcopy(data_dict['rects'])
    liveness = data_dict['liveness']

    img_big = np.zeros((1080, 1920, 3))
    try:
        img_big[:img_h, :img_w] = img
    except:
        assert False, 'jumping because paste fail: {}'.format(data_dict['img_path'])

    img_big = img_big.astype(np.uint8)
    img_big = bgr2yuv444(img_big)
    bbox = [int(x) for x in rects]

    cropped_imgs, bbox_lists, layer = img_pyramid_resizer(img=img_big, rois=[bbox], norm_len=int(128.0 / expand_ratio),
                                                          input_hw=[128, 128],
                                                          use_pyramid=True, use_resizer=True, im_w=img_w, im_h=img_h,
                                                          restrict_bound=True, norm_method='sqrt_area',
                                                          platform='x2')

    image = cropped_imgs[0]

    return image, liveness


def parse_args():
    parser = argparse.ArgumentParser(description='Predict')
    parser.add_argument('--file', help='image info list file.', required=False, type=str)
    parser.add_argument('--dir', help='image dir.', required=False, type=str)
    parser.add_argument('--save', help='save result file.', required=False, type=str)
    parser.add_argument('--prefix', help='model prefix.', required=True, type=str)
    parser.add_argument('--epoch', help='model epoch.', required=True, type=int)
    parser.add_argument('--gpus', help='which gpus to predict.', required=True, type=str, default='-1')
    parser.add_argument('--batch-size', help='predict batch size.', required=True, type=int, default=1)
    parser.add_argument('--input-shape', help='input image shape (HWC), eg. \'128,128,1\'.', required=True, type=str,
                        default='128,128,1')
    parser.add_argument('--input-format', help='input image format (rgb, gray, yuv), eg. \'gray\'.', required=True,
                        type=str, default='gray')
    parser.add_argument('--data-type', help='input data type', required=True, type=int, default=1)
    parser.add_argument('--expand-ratio', help='set expand ratio', default=1.0, type=float)
    parser.add_argument('--thresh', help='thresh', required=False, type=float, default=0.5)
    parser.add_argument('--is-qnn', help='whether to use qnn predict.', required=False, type=int, default=0)
    parser.add_argument('--save-feature', help='whether to save feature for visualization', required=False, type=str,
                        default=False)
    parser.add_argument('--save-badcase', help='whether to save badcase', required=False, type=str, default=False)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.data_type == 1:
        process_method = process_method_1
        assert args.input_format == 'gray'
    elif args.data_type == 2:
        process_method = process_method_2
    elif args.data_type == 3:
        process_method = process_method_3
    elif args.data_type == 4:
        process_method = process_method_4
    elif args.data_type == 5:
        process_method = process_method_5
        assert args.input_format == 'gray'
    elif args.data_type == 6:
        process_method = process_method_6
        assert args.input_format == 'gray'
    elif args.data_type == 7:
        process_method = process_method_7
        assert args.input_format == 'yuv'
    else:
        assert False, 'not support data type: {}'.format(args.data_type)

    anti_spoofing_predictor = AntiSpoofingPredictor(args, process_method)
    anti_spoofing_predictor.predict()
    anti_spoofing_predictor.evaluate()
    anti_spoofing_predictor.evaluate_multi_thresh()

