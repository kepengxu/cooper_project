# -*- coding: UTF-8 -*-
import os.path as osp
import sys
dirpath=osp.dirname(__file__)
def add(path):
    if path not in sys.path:
        sys.path.insert(0,path)