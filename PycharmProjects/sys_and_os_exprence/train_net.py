# coding=utf-8 #有中文注释，记得加上这个
#!/usr/bin/env python
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Faster R-CNN network on a region of interest database."""

#首先，加载进来需要的各种模块。
#python中，每个py文件被称之为模块，，每个具有__init__.py文件的目录被称为包
import _init_paths #_init_paths是一个.py文件，用来设置Faster-RCNN的路径
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg,cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
from networks.factory import get_network
import argparse
import pprint
import numpy as np
import sys
import os

def parse_args():
    """
    Parse input arguments解析输入参数
    这里的参数指的是，在运行train_net.py这个文件时，需要的输入参数
    """
    parser = argparse.ArgumentParser(description='Train a Faster R-CNN network')
    parser.add_argument('--device', dest='device', help='device to use',
                        default='cpu', type=str)  #--device代表选用cpu还是gpu,默认cpu
    parser.add_argument('--device_id', dest='device_id', help='device id to use',
                        default=0, type=int) #--device_id代表机器上的cpu或者gpu的编号
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)  # --solver代表模型的配置文件，这个参数的值就固定是VGG_CNN_M_1024
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=70000, type=int) #--iters代表训练时的最大迭代步数，默认是70000步
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)  #--weights代表权重文件，也就是预训练好的模型。
    #这里用的是Imagenet上预训练好的模型VGG_imagenet.npy，
    #存放在目录Faster-RCNN_TF/data/pretrain_model下
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='kitti_train', type=str)  #--imdb代表训练数据库的名称，默认是kitti_train。
    #该工程中，提供了5种数据库来训练网络，并分别给出了各自的数据读写接口，
    #5种数据库分别是pascal_voc，coco，kitti，nissan，nthu
    #（工程中，说是提供了5种数据库，但是也就只给出了各自的数据库读写接口，并没有给出实际的数据库，所以得需要自己另行下载，工程中没有提供）。
    #另外，这个数据库名称是固定的，该名称在Faster-RCNN_TF/lib/datasets/factory.py中
    #被定义了具体的格式：以pascal_voc数据库为例，参数--imdb的值应为voc_2007_train。
    #文件factory.py会在下一篇做更进一步的解释
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default='kitti_train', type=str) #--network代表选择训练网络还是测试网络，
    #这个参数的值的形式是固定的，必须是kitti_train的形式，
    #前半部分kitti可以随便定义（但是不能有下划线），后半部分必须是_train
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__': #主函数

    args = parse_args()
    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg) #cfg就是Faster-RCNN_TF/lib/fast_rcnn/config.py,
    #是网络训练的参数文件。这里的参数指的是网络在训练过程需要用到的各种参数。

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)

    #加载训练数据。函数get_imdb在Faster-RCNN/lib/datasetes/factory.py中被定义。
    imdb = get_imdb(args.imdb_name)
    print 'Loaded dataset `{:s}` for training'.format(imdb.name)

    #将训练数据变成minibatch的形式。
    #函数get_training_roidb在Faster-RCNN/lib/fast_rcnn/train.py中被定义
    roidb = get_training_roidb(imdb)

    #设置保存（训练好的模型）的目录。如果该目录没有，会自动新建一个。
    #函数get_output_dir在Faster-RCNN_TF/lib/fast_rcnn/config.py中被定义
    output_dir = get_output_dir(imdb, None)
    print 'Output will be saved to `{:s}`'.format(output_dir)

    #设置GPU或者CPU的 id
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)
    device_name = '/{}:{:d}'.format(args.device,args.device_id)
    print device_name

    #按照args.network_name获取网络。选择train网络或者test网络。
    #为什么参数args.network_name的值有固定的格式，看函数get_network就知道了。
    #函数get_network在Faster-RCNN_TF/lib/networks/factory.py中被定义。
    network = get_network(args.network_name)
    print 'Use network `{:s}` in training'.format(args.network_name)

    #启动Faster-RCNN网络训练。
    #函数train_net在Faster-RCNN_TF/lib/fast_rcnn/train.py中被定义
    train_net(network, imdb, roidb, output_dir,
              pretrained_model=args.pretrained_model,
              max_iters=args.max_iters)