# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import os
from utils.pyramid_resizer.lib_utils import *
import ctypes
import numpy as np


original_dir = os.path.dirname(os.path.abspath(os.path.expanduser(os.readlink(__file__)))) if os.path.islink(__file__) else os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))

pyramid_path = '%s' % original_dir
lib_path = '%s/_pyramid.so' % pyramid_path
_LIB = load_lib(lib_path)

_IMG_TYPE = {
  'yuv': 0, 
  'yuv_i420': 1,
  'yuv_nv12': 2,
  'yuv_nv21': 3,
  'y_only': 4
}

_PYRAMID_TYPE = {
  'matrix': 0,
  'x1': 1
}

class Pyramid:
  def __init__(self, target, img_type, src_h, src_w):
    if target == 'matrix':
      self.config_file = '%s/pyramid_matrix.cfg' % pyramid_path
    elif target == 'x1':
      self.config_file = '%s/pyramid_x1.cfg' % pyramid_path

    handle = HANDLE()
    img_type = _IMG_TYPE[img_type]
    pyramid_type = _PYRAMID_TYPE[target]
    ret = _LIB.PyramidCreate(INT(pyramid_type), c_str(self.config_file), INT(img_type), INT(src_h), INT(src_w), ctypes.byref(handle))
    if ret != 0:
      self.handle = None
      raise Exception('call PyramidCreate failed.')
    else:
      self.handle = handle

  def do_pyramid(self, img, src_h, src_w):
    '''
      do pyramid resize for image
      the image is in yuv format, so need src_h and src_w to get image size
      return numpy list for all resized image
    '''
    # call do run pyramid
    ret = _LIB.PyramidScale(self.handle, img.ctypes.data_as(INT8_p), INT(src_h), INT(src_w), BOOL(False))
    if ret != 0:
      print('call pyramid scale failed')
      return ret, []

    # get out image shapes
    num_layers = INT()
    ds_h = ctypes.POINTER(INT)()
    ds_w = ctypes.POINTER(INT)()
    channels = INT()
    ret = _LIB.PyramidGetDSLayerHW(self.handle, ctypes.byref(ds_h), ctypes.byref(ds_w), ctypes.byref(num_layers), ctypes.byref(channels))
    if ret != 0:
      print('call get ds layer HW failed')
      return ret, []
    # alloc numpy data for layer output
    output_list = []
    result_list = []
    num_layers = num_layers.value
    for i in range(num_layers):
      if channels.value == 3:
        out = np.empty((ds_h[i], ds_w[i], 3), dtype=np.uint8)
      else:
        out = np.empty((ds_h[i], ds_w[i]), dtype=np.uint8)
      output_list.append(out)
      result_list.append(out.ctypes.data_as(INT8_p))
    # get output from pyramid
    ret = _LIB.PyramidGetDSLayerOutput(self.handle, c_array(INT8_p, result_list))
    return ret, output_list 

  def run(self, img_data, src_h, src_w):
    '''
      img_file_name : input image file name that will be resize to pyramid images, for [origin, 1/2, 1/4, 1/8, ...]
    '''
    # run pyramid resize command
    ret, out_list = self.do_pyramid(img_data, src_h, src_w)
    if ret != 0:
      print('run pyramid failed')
      return False, []

    #print('pyramid resize done, get resized ', len(out_list))
    return True, out_list

if __name__ == '__main__':
  #test for pyramid 
  #target = 'matrix'
  target = 'x1'
  #img_path = '/home/users/kai.lu/code/simulator/test_img_2.jpg'
  img_path = '/home/users/kai.lu/video-00000009.jpg'
  img_data = cv2.imread(img_path, 1)
  print('image shape:', img_data.shape)
  src_h, src_w, _ = img_data.shape
  # convert bgr to yuv
  img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
  print('img gray shape:', img_data.shape)
  
  pyramid_op = Pyramid(target, 'y_only', src_h, src_w)
  ret, out_list = pyramid_op.run(img_data, src_h, src_w)
  if not ret:
    raise Exception('here do pyramid resize failed')

  print('here get pyramid resize result len: ', len(out_list))

  #save output data to png file
  out_index = 0
  for img in out_list:
    print('img shape:', img.shape)
    #img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_I420)
    out_file_name = './logs/ds_img%d.png' % out_index
    cv2.imwrite(out_file_name, img)
    out_index += 1

  print('dump pyramid resize to png done.')
