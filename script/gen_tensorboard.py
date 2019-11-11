from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
import numpy as np
import mxnet as mx
import pickle
import os
from mxboard import SummaryWriter
import random
import copy
import logging
from utils.base_config import get_config
from utils.utils import parse_args
from script.base_gen import base_gen
import json
np.set_printoptions(threshold=np.inf)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


root_dir = os.path.abspath(__file__).split('script')[0]
sys.path.insert(0, root_dir + '/script')

class gen(base_gen):
    def __init__(self, args):
        super(gen, self).__init__(args)
        self.parse_file_dir()

    def run(self):
        resized_images = mx.nd.load('./tmps/resized_images.ndarray')[0]
        convnet_codes = mx.nd.load('./tmps/convnet_codes.ndarray')[0]
        labels = mx.nd.load('./tmps/labels.ndarray')[0]
        pkl_file = open('./tmps/tag', 'rb')
        data1 = pickle.load(pkl_file)

        list_index = [i for i in range(1, len(data1))]

        data2 = []
        k2 = []
       # if int(self.config.function.shownum) != 0:
      #  if True:
        print('here')
        anlist = []
        polist = []
        for i in list_index:
            if labels[i] == 1:
                anlist.append(i)
            else:
                polist.append(i)
        if not os.path.exists(self.config.function.keys_path):
            logging.info('Saving keys to keys.json')
            keys = random.sample(anlist, min(self.config.function.shownum, len(anlist))) + random.sample(polist,
                                            min(self.config.function.shownum, len(polist)))
            keys.sort()
            with open(self.config.function.keys_path, 'w') as f:
                json.dump(keys, f)
            logging.info('Successfully saving keys')
        else:
            logging.info('Loading keys.json')
            with open(self.config.function.keys_path, 'r') as f:
            	keys = json.load(f)
            logging.info('Successfully loading keys')
        resized_images = resized_images[keys]
        convnet_codes = convnet_codes[keys]
        labels = labels[keys].asnumpy()
        for i in keys:
            data2.append(copy.copy(data1[i]))
     #   else:
     #       labels = labels.asnumpy()
     #       data2 = data1
        print (convnet_codes.shape,len(data2),labels.shape,resized_images.shape)
        with SummaryWriter(logdir=self.config.function.location) as sw:
            sw.add_image(tag=self.config.function.tagname, image=resized_images)
            # print (convnet_codes.shape,len(data2),labels.shape,resized_images.shape)
            sw.add_embedding(tag=self.config.function.tagname, embedding=convnet_codes, images=resized_images,
                             labels=labels)

        # print ('Index\tLabel\n')
        with open(os.path.join(self.config.function.location, self.config.function.tagname, 'metadata.tsv'), 'w') as f:
            f.write('Path\tLabel\n')
            for path, label in zip(data2, labels):
                f.write('{}\t{}\n'.format(path, label))


def main():
    args = parse_args()
    # print(args.file, args.number, args.save)
    config = get_config(args)
    gen_example = gen(config)
    gen_example.run()
    command = 'rm -r ./tmps'
    os.system(command)


if __name__ == '__main__':
    main()
