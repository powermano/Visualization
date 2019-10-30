# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import random

'''
data type info:
    1, use pyramid and resizer.
    2, not resize img, only crop.
    3, not use pyramid and resizer.
    4, not use pyramid and resizer, not deformation.
    5, use pyramid and resizer, not deformation.
    6, not use pyramid, bug use resizer, not deformation.
    7, yuv, use pyramid and resizer, not deformation.
'''

if __name__ == '__main__':
    is_parallel_run = True
    save_feature = '0'
    save_badcase = '0'

    test_type = 'X1600_v2'
    test_file = '/home/users/tao.cai/aiot_face_anti_spoofing_tools/predict/test_list/{}_test.txt'.format(
        test_type)
    save_file = './anti_spoofing_outputs.txt'
    gpus = '0'
    batch_size = '64'
    input_shape = '128,128,3'
    input_format = 'rgb'
    is_qnn = '0'

    model_dir = '/home/users/tao.cai/Workspace/anti_spoof/gluonface/0815/triplet_loss/margin-0.2/awl/'
    model_list = ['gpu']
    epoch_list = [10]
    data_type_list = [4]
    expand_ratio_list = [1.5]

    for model, epoch, data_type, expand_ratio in zip(model_list, epoch_list, data_type_list, expand_ratio_list):
        if is_qnn == '1':
            prefix = model_dir + 'model-absorb-bn-ft-hobot-predict-{}'.format(str(model))
            work_dir = './results/{}/result_qnn_model{}_epoch{}_dataType{}_expandRatio{}'.format(
                test_type, model, epoch, data_type, expand_ratio)
        else:
            prefix = model_dir + 'model-{}'.format(str(model))
            work_dir = './results/{}/python3.4.5_0815_triplet_loss_fix_margin-0.2_awl_feature_gpu_model{}_epoch{}_dataType{}_expandRatio{}'.format(
                        test_type, model, epoch, data_type, expand_ratio)
          #  work_dir = './results/{}/python3.4.5_0815_wobn_triplet_loss_margin-0.2_l2_wo_sum_feature_gpu_model{}_epoch{}_dataType{}_expandRatio{}'.format(
          #               test_type, model, epoch, data_type, expand_ratio)
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        os.chdir(work_dir)
        now_dir = os.getcwd()
        print('work in ', os.getcwd())
        command = 'sh {}/../../../predict.sh {} {} {} {} {} {} {} {} {} {} {} {} {} {}'.format(now_dir, test_file,
                                                                                         save_file,
                                                                                         prefix,
                                                                                         epoch,
                                                                                         '-1' if gpus == '-1' else random.randint(
                                                                                             0, 3),
                                                                                         batch_size,
                                                                                         input_shape,
                                                                                         input_format,
                                                                                         data_type,
                                                                                         expand_ratio,
                                                                                         is_qnn,
                                                                                         now_dir,
                                                                                         save_feature,
                                                                                         save_badcase)
        print(command)
        if not is_parallel_run:
            os.system(command)
        else:
            os.system('nohup {}  &'.format(command))
        os.chdir('../../../')
        print('work in ', os.getcwd())
