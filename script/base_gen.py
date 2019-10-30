from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import cv2
import numpy as np

import argparse
from tqdm import tqdm
import mxnet as mx
import cv2
import pickle
import os
from mxboard import SummaryWriter
import random
import copy
np.set_printoptions(threshold=np.inf)
import mxnet as mx
import random
import time
import copy
import math
import argparse
import logging
import shutil


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


root_dir = os.path.abspath(__file__).split('script')[0]
sys.path.insert(0, root_dir + '/script')


class base_gen(object):
    def __init__(self, args):
        self.config = args

    def parse_file_dir(self):
        with open(self.config.input.file_path, 'r') as f:
            lines = [x.strip() for x in f.readlines()]

        length = len(lines)
        vecstart, vecend = list(map(int, self.config.input.vector.split(',')))
      #  tagname = 'face_anti-spoofing'

      #  location = './logs'
        rois = list(map(int, self.config.input.roi.split(',')))

        convnet_codes = None  # N * 1000
        resized_images = None  # NCHW
        labels = []
        tag = []
        resized_images = mx.nd.ones((length, 3, self.config.function.resize, self.config.function.resize), dtype='uint8')
        convnet_codes = mx.nd.ones((length, vecend - vecstart + 1), dtype='uint8')
        labels = mx.nd.ones(length, dtype='uint8')
        if not self.config.function.show_image:
            black_square = np.zeros((self.config.function.resize, self.config.function.resize, 3))
            black_square = mx.nd.array(black_square).transpose((2, 0, 1)).expand_dims(axis=0).astype('uint8')
            red_square = np.zeros((self.config.function.resize, self.config.function.resize, 3))
            red_square[:, :, 0] = 255
            red_square = mx.nd.array(red_square).transpose((2, 0, 1)).expand_dims(axis=0).astype('uint8')
        for i in tqdm(range(length)):
            t = lines[i].split()
            labels[i] = int(float(t[self.config.input.label]))
            tag.append(t[self.config.input.picture_path])
            if self.config.function.noimage == 0:
                if self.config.function.show_image:
                    r = list(map(int,list(map(float, t[rois[0]:rois[1]+1]))))
                    img = cv2.imread(t[self.config.input.picture_path])[int(float(r[1])):int(float(r[3])),
                          int(float(r[0])):int(float(r[2]))]
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                    resized_image = cv2.resize(img, (self.config.input.resize, self.config.input.resize),
                                               interpolation=cv2.INTER_AREA)
                    resized_image = mx.nd.array(resized_image)
                    resized_image = resized_image.transpose((2, 0, 1)).expand_dims(axis=0).astype('uint8')
                else:
                    if labels[i] == 1:
                        resized_image = red_square
                    else:
                        resized_image = black_square
                resized_images[i] = resized_image
            s = list(map(int, t[vecstart:vecend + 1]))

            # convnet_codes[i] = mx.nd.array([s]).astype('uint8')
            convnet_codes[i] = mx.nd.array([s])

        if self.config.function.save_tmp:
            if not os.path.exists('./tmps'):
                os.mkdir('tmps')
            mx.nd.save('./tmps/convnet_codes.ndarray', convnet_codes)
            mx.nd.save('./tmps/resized_images.ndarray', resized_images)
            mx.nd.save('./tmps/labels.ndarray', mx.nd.array(labels).astype('uint8'))
            with open('./tmps/tag', 'wb') as l:
                pickle.dump(tag, l, protocol=2)
