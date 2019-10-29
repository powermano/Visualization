# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import ctypes

def load_lib(lib_path):
  lib = ctypes.cdll.LoadLibrary(lib_path)
  return lib

INT8_p = ctypes.POINTER(ctypes.c_uint8)
INT = ctypes.c_int
BOOL = ctypes.c_bool
HANDLE = ctypes.c_void_p

def c_str(string):
  """"Convert a python string to C string."""
  return ctypes.c_char_p(string.encode('utf-8'))

def c_array(ctype, values):
    """Create ctypes array from a python array."""
    return (ctype * len(values))(*values)
