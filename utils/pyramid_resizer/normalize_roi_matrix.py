from __future__ import print_function
import numpy as np

def get_normalize_roi(pyramid_img_list, bbox_list, norm_len, dst_shape, norm_method, im_w, im_h, restrict_bound, **kwargs):
    '''
    @breif normalize bbox by norm_len
           bbox_list: [[x1, y1, x2, y2]...]
           norm_len : int
      search for the max suitable pyramid img for dst-shape
      i.e. 
        dst-shape is (h,w)
        norm_dst = norm_method(norm_method)
        roi_norm_len = norm_method(roi)
        if roi_norm_len / 2 < norm_dst then choose img0, for there need no scale from origin-img
        if roi_norm_len / 4 < norm_dst then choose img4, use the 1/2 origin-img
        if roi_norm_len / 8 < norm_dst then choose img8, use the 1/4 origin-img
        if roi_norm_len / 16 < norm_dst then choose img12, use the 1/8 origin img
        if roi_norm_len / 32 < norm_dst then choose img16, use the 1/16 origin img
      
      the condition can be transformed to roi_norm_len < norm_dst * 2
    ''' 
    def _norm_func(h, w):
      if norm_method == 'width':
        return w
      elif norm_method == 'height':
        return h
      elif norm_method == 'longside':
        return max(h, w)
      elif norm_method == 'sqrt_area' or norm_method == 'sqrtarea':
        return np.sqrt(h**2 + w**2)
      elif norm_method == 'none':
        return max(h, w)

    roi_regions = []
    dst_h = dst_shape[0]
    dst_w = dst_shape[1]

    for bbox in bbox_list:
      if norm_method == 'none':
        left = int(bbox[0])
        top = int(bbox[1])
        right = int(bbox[2])
        bottom = int(bbox[3])
      else:
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        center_x = int(bbox[0] + width / 2)
        center_y = int(bbox[1] + height / 2)
        bbox_norm_len = int(_norm_func(height, width))
        crop_w = int(dst_w * bbox_norm_len / norm_len / 2)
        crop_h = int(dst_h * bbox_norm_len / norm_len / 2)
        left = center_x - crop_w
        top = center_y - crop_h
        right = center_x + crop_w
        bottom = center_y + crop_h

      if restrict_bound:
        left = max(left, 0)
        top = max(top, 0)
        right = min(right, im_w - 1)
        bottom = min(bottom, im_h - 1)
      
      roi_regions.append((left, top, right, bottom))

    # search pyramid image
    norm_len_dst = int(_norm_func(dst_h, dst_w))
    res_img_list = []
    res_roi_list = []
    for roi in roi_regions:
      x1, y1, x2, y2  = roi
      roi_norm_len = int(_norm_func(y2 - y1, x2 - x1))
      i = 0
      exp = 2
      while (i < 5):
        if roi_norm_len < (norm_len_dst * exp):
          x11 = x1 & (~0x1)
          y11 = y1 & (~0x1)
          x22 = (x2 - 1) | (0x1)
          y22 = (y2 - 1) | (0x1)

          h = y22 - y11 + 1
          w = x22 - x11 + 1
          if (h >= (dst_h * 2) or (w >= (dst_w * 2))):
            exp = exp * 2
            i = i + 1
            x1, y1, x2, y2 = x1/2, y1/2, x2/2, y2/2
            continue

          res_roi_list.append((x11, y11, x22, y22))
          res_img_list.append(pyramid_img_list[i * 4])
          break
        exp = exp * 2
        i = i + 1
        x1, y1, x2, y2 = x1/2, y1/2, x2/2, y2/2

    return roi_regions, res_img_list, res_roi_list

