# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from utils.pyramid_resizer.lib_utils import *
import numpy as np
import cv2


original_dir = os.path.dirname(os.path.abspath(os.path.expanduser(os.readlink(__file__)))) if os.path.islink(__file__) else os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))

resizer_path = '%s' % original_dir
lib_path = '%s/_resizer.so' % resizer_path
_LIB = load_lib(lib_path)

_IMG_TYPE = {
  'yuv': 0, 
  'yuv_i420': 1,
  'yuv_nv12': 2,
  'yuv_nv21': 3,
  'y_only': 4
}

_RESIZE_TYPE = {
  'matrix': 0,
  'x1': 1
}

class Resizer:
  def __init__(self, **kwargs):
    self.img_type = kwargs['img_type'] if 'img_type' in kwargs else 'yuv_i420'
    self.dst_h = kwargs['dst_h']
    self.dst_w = kwargs['dst_w']
    # padding mode not use False
    self.padding_mod = kwargs['padding_mod'] if 'padding_mod' in kwargs else True
    self.debug = kwargs['debug'] if 'debug' in kwargs else False
    resize_type = kwargs['target'] if 'target' in kwargs else 'matrix'
    self.resize_type = _RESIZE_TYPE[resize_type]
    if self.img_type == 'y_only':
      self.output_shape = (self.dst_h, self.dst_w)
    elif self.img_type != 'yuv':
      uv_dst_h = self.dst_h / 2
      uv_dst_w = self.dst_w / 2
      self.output_shape = (uv_dst_h * 3, uv_dst_w * 2)
    else:
      self.output_shape = (self.dst_h, self.dst_w, 3)

  def _call_resize(self, img, img_type, img_src_h, img_src_w,
                   left, top, right, bottom):
    assert(isinstance(img, np.ndarray))
    img_data = np.ascontiguousarray(img)
    img_type = _IMG_TYPE[img_type]
    out_data = np.empty(self.output_shape, dtype=np.uint8)
    ret = _LIB.Resize(img_data.ctypes.data_as(INT8_p), INT(self.resize_type), INT(img_type),
                      INT(img_src_h), INT(img_src_w),
                      INT(left), INT(top), INT(right), INT(bottom),
                      INT(self.dst_h), INT(self.dst_w), BOOL(self.padding_mod), BOOL(self.debug),
                      out_data.ctypes.data_as(INT8_p))
    return ret, out_data

  def run(self, img_list, roi_list):
    resize_img_list = []
    for img, roi_reg in zip(img_list, roi_list):
      #print('resize scale: ', scale)
      #print('resize img with shape: ', img.shape)
      ret, out_data = self._call_resize(img, self.img_type, int(img.shape[0]), int(img.shape[1]),
                         int(roi_reg[0]), int(roi_reg[1]), int(roi_reg[2]), int(roi_reg[3]))
      if ret != 0:
        raise Exception('execute resizer failed')

      resize_img_list.append(out_data)

    return resize_img_list

if __name__ == '__main__':
  # test for Resizer
  img_path = '/home/users/kai.lu/code/simulator/test_img_2.jpg'
  l, t, r, b = (1141, 325, 1280, 460) 
  dst_h, dst_w = (128, 128)
  img = cv2.imread(img_path, 1)
  src_h, src_w, _ = img.shape
  print('src shape: ', img.shape)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  print('yuv shape: ', img.shape)
  resizer = Resizer(target='matrix', img_type='y_only', dst_h=dst_h, dst_w=dst_w, debug=False)
  result = resizer.run([img,], [(l,t,r,b),]) 
  print('get resize result: ', len(result), ' with shape: ', result[0].shape)
  out_img = result[0]
  out_img = cv2.cvtColor(out_img, cv2.COLOR_GRAY2BGR)
  cv2.imwrite('output.png', out_img)


