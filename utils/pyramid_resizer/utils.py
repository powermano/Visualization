# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import cv2
import logging

from utils.pyramid_resizer.pyramid import Pyramid
from utils.pyramid_resizer.resizer import Resizer


def limit_roi(roi, im_height, im_width):
    l = max(0, roi[0])
    t = max(0, roi[1])
    r = min(im_width - 1, roi[2])
    b = min(im_height - 1, roi[3])
    return [l, t, r, b]

def img_crop(im, roi):
    roi = [int(v) for v in roi]
    cut_roi = limit_roi(roi, im.shape[0], im.shape[1])

    if len(im.shape) == 3:
        im_roi = im[cut_roi[1]:cut_roi[3]+1, cut_roi[0]:cut_roi[2]+1, :]
    else:
        im_roi = im[cut_roi[1]:cut_roi[3]+1, cut_roi[0]:cut_roi[2]+1]
    im_roi = cv2.copyMakeBorder(im_roi,
                                cut_roi[1] - roi[1], roi[3] - cut_roi[3],
                                cut_roi[0] - roi[0], roi[2] - cut_roi[2],
                                cv2.BORDER_CONSTANT)
    im_roi = im_roi.astype(np.float32)
    return im_roi

def get_roi_regions_from_rois(rois, norm_len, input_hw):
    roi_regions = []
    for roi in rois:
        center = ((roi[0] + roi[2]) / 2.0, (roi[1] + roi[3]) / 2.0)
        crop_scale = (roi[3] - roi[1]) / float(norm_len)
        crop_wh = (crop_scale * input_hw[1], crop_scale * input_hw[0])
        roi_regions.append([center[0] - crop_wh[0] / 2.0, center[1] - crop_wh[1] / 2.0,
                            center[0] + crop_wh[0] / 2.0, center[1] + crop_wh[1] / 2.0])
    return roi_regions

def img_pyramid_resizer(img, rois, norm_len, input_hw, use_pyramid, use_resizer,
                    im_w, im_h, restrict_bound, norm_method, interpolation=cv2.INTER_LINEAR, platform='x1'):
    if platform == 'x1':
        from normalize_roi_x1 import get_normalize_roi
        pyramid = Pyramid(target='x1', img_type='y_only', src_h=1080, src_w=1920)
        resizer = Resizer(target='x1', img_type='y_only', dst_h=input_hw[0], dst_w=input_hw[1])
    elif platform == 'x2' or platform == 'matrix' or platform == 'xforce':
        from normalize_roi_matrix import get_normalize_roi
        pyramid = Pyramid(target='matrix', img_type='yuv', src_h=1080, src_w=1920)
        resizer = Resizer(target='matrix', img_type='yuv', dst_h=input_hw[0], dst_w=input_hw[1])
    else:
        assert False, 'not support platform: {}'.format(platform)

    if use_resizer:
        if use_pyramid:
            # use pyramid and resizer
            ret, pyramid_img_list = pyramid.run(img, img.shape[0], img.shape[1])
            assert ret
        else:
            pyramid_img_list = [img]

        roi_regions, img_list, roi_list = get_normalize_roi(pyramid_img_list=pyramid_img_list,
                                                            bbox_list=rois,
                                                            norm_len=norm_len,
                                                            dst_shape=input_hw,
                                                            norm_method=norm_method,
                                                            im_w=im_w,
                                                            im_h=im_h,
                                                            restrict_bound=restrict_bound)

        cropped_img_list = resizer.run(img_list, roi_list)
        return cropped_img_list, roi_regions, []
    else:
        roi_list = get_roi_regions_from_rois(rois=rois, norm_len=norm_len, input_hw=input_hw)
        cropped_img_list = []
        for roi_bbox in roi_list:
            roi_patch = img_crop(img, roi_bbox)
            cropped_img = cv2.resize(roi_patch, dsize=(input_hw[1], input_hw[0]), interpolation=interpolation)
            cropped_img_list.append(cropped_img)
        return cropped_img_list, roi_list, []
