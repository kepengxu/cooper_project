# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
import PIL.Image as Image
import time
import cv2 as cv
import os
def read_dir_picture(dir_path,NUM_CLASSES,image_h,image_w,transform_to_gray=True):
    '''
    tips:文件夹内文件开头种类一定要小于  NUM_CLASSES
    this function will return two parmater:
        result_example:[epoch_size,image_h*image_w]
        onehot [epoch,num_classes]

        the rule of image class is image first name[0]
    need to import this model
    import tensorflow as tf
    import numpy as np
    import PIL.Image as Image
    import time
    import cv2 as cv
    import os

    :param dir_path:
    :param num_class:
    :param image_h:
    :param image_w:
    :return:

    '''
    file_list = os.listdir(dir_path)
    n = file_list.__len__()  # 样本个数
    result_example = np.array([[]])
    time_reslut_onehot = np.array([[]])
    # print "result",result.shape
    i = 1
    for name in file_list:
        #    int(name[0])
        image = Image.open(dir_path + '/' + name).convert('L')
        #    image=np.expand_dims(image)
        image_arr = np.array(image).reshape(image_h*image_w)
        image_arr = np.expand_dims(image_arr, 0)
        #    print image_arr.shape
        if i == 1:
            time_reslut_onehot = np.array([[int(name[0])]])
            result_example = image_arr
            i = 0
        else:
            time_reslut_onehot = np.row_stack((time_reslut_onehot, name[0]))
            result_example = np.row_stack((result_example, image_arr))
    print result_example.shape
    # the size of image 2073600
    indices = tf.expand_dims(tf.range(0, n, ), 1)
    concated = tf.concat([indices, time_reslut_onehot], 1)
    onehot_labels = tf.sparse_to_dense(concated, [n, NUM_CLASSES], 1.0, 0.0)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        onehot = sess.run(onehot_labels)
        num=np.column_stack((onehot,result_example))
        np.random.shuffle(num)
        print "the num shape",num.shape
        onehot=num[:,0:NUM_CLASSES]
        result_example=num[:,NUM_CLASSES:]
        return result_example,onehot

#under this is example in my
dirpath='/home/cooper/图片/image_209'
ex,one=read_dir_picture(dirpath,9,1920,1080,transform_to_gray=True)
print "function is OK"
print ex.shape
print one