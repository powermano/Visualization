from __future__ import print_function

def _calc_r_b(norm_len):
  bits = 0
  len = norm_len * 0xFF
  while len != 0:
    len = len >> 1
    bits = bits + 1
  bits = bits - 1
  b = bits & 0xF
  r = ((1 << bits) / norm_len) & 0xFF
  return r, b

def _calc_cnn_roi(bbox, norm_method, r, b, dst_h, dst_w):
  l = int(bbox[0])
  t = int(bbox[1])
  w = int(bbox[2] - bbox[0])
  h = int(bbox[3] - bbox[1])
  center_x = int((l + w - 1 + l) >> 1)
  center_y = int((t + h - 1 + t) >> 1)
  scale_r = 0
  if norm_method == 'none':
    crop_w = w
    crop_h = h
  elif norm_method == 'width':
    scale_r = w * r
    a = (dst_h << 4) / dst_w
    crop_w = (scale_r * dst_w) >> b
    crop_w = (crop_w + 1) & (~0x1) # make it even
    crop_h = (crop_w * ((dst_h << 4) / dst_w)) >> 4
    crop_h = (crop_h + 1) & (~0x1)
  elif norm_method == 'height':
    scale_r = h * r
    crop_h = (scale_r * dst_h) >> b
    crop_h = (crop_h + 1) & (~0x1) # make it even
    crop_w = (crop_h * ((dst_w << 4) / dst_h)) >> 4
    crop_w = (crop_w + 1) & (~0x1)
  elif norm_method == 'longside':
    if w > h:
      scale_r = w * r
      crop_w = (scale_r * dst_w) >> b
      crop_w = (crop_w + 1) & (~0x1) # make it even
      crop_h = h + (crop_w - w)
      crop_h = (crop_h + 1) & (~0x1)
    else:
      scale_r = h * r
      crop_h = (scale_r * dst_h) >> b
      crop_h = (crop_h + 1) & (~0x1) # make it even
      crop_w = w + crop_h - h
      crop_w = (crop_w + 1) & (~0x1)
  elif norm_method == 'sqrt_area' or norm_method == 'sqrtarea':
    if w > h:
      scale_r = w * r
      crop_w = (scale_r * dst_w) >> b
      crop_w = (crop_w + 1) & (~0x1) # make it even
      crop_h = crop_w
    else:
      scale_r = h * r
      crop_h = (scale_r * dst_h) >> b
      crop_h = (crop_h + 1) & (~0x1) # make it even
      crop_w = crop_h
  else:
    assert False, 'not support norm method'

  left = center_x - (crop_w >> 1)
  top = center_y - (crop_h >> 1)
  right = left + crop_w - 1
  bottom = top + crop_h - 1

  return scale_r, (left, top, right, bottom)

def get_normalize_roi(pyramid_img_list, bbox_list, norm_len, dst_shape, norm_method, im_w, im_h, restrict_bound):
    '''
    @breif normalize bbox by norm_len
           bbox_list: [[x1, y1, x2, y2]...]
           norm_len : int
           dst_shape : (h,w)
           bbox_norm_func:  function that tells how to get norm_length of the bbox. 
    ''' 
    # if x1/x2's scaldownFact is 4, then scaldownFact should be 3 (4 - 1).
    # this param(scaldownFact) in pyramid_x1.cfg also should be 3 (4 - 1).
    scaldownFact = 3  # must same as "ScaleFact" config in pyramid_x1.cfg
    SN = 16
    dst_h = dst_shape[0]
    dst_w = dst_shape[1]
    r, b = _calc_r_b(norm_len)
    roi_regions = []
    res_img_list = []
    res_roi_list = []

    for bbox in bbox_list:
      scale_r, roi = _calc_cnn_roi(bbox, norm_method, r, b, dst_h, dst_w)
      if b > 6:
        scale_r = scale_r >> (b - 6)
      else:
        scale_r = scale_r << (6 - b)

      if restrict_bound:
        roi = [max(roi[0], 0), max(roi[1], 0), min(roi[2], im_w - 1), min(roi[3], im_h - 1)]
      roi_regions.append(roi)
      x1, y1, x2, y2 = roi

      i = 0
      layer_scale = 1 << 6
      layer_scale = (layer_scale * (16 + scaldownFact + 1) + 8) >> 4
      coord_scale = 1 << 15
      while i < len(pyramid_img_list):
        if scale_r < layer_scale:
          x1 = x1 * coord_scale >> 15
          y1 = y1 * coord_scale >> 15
          x2 = x2 * coord_scale >> 15
          y2 = y2 * coord_scale >> 15
          res_roi_list.append((x1, y1, x2, y2))
          res_img_list.append(pyramid_img_list[i])
          break
        i = i + 1
        coord_scale = coord_scale * SN / (SN + scaldownFact + 1)
        layer_scale = (layer_scale * (16 + scaldownFact + 1) + 8) >> 4

    return roi_regions, res_img_list, res_roi_list

