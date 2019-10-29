"""Base Config."""
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import yaml
import logging
from easydict import EasyDict


def get_config(args):
    """
    Generate config.

    """

    # 1. gennerate config based on yaml file
    with open(args.cfg, 'r') as f:
        if sys.version_info.major == 3:
            config = EasyDict(yaml.load(f.read(), Loader=yaml.FullLoader))
        else:
            config = EasyDict(yaml.load(f.read()))

    return config
