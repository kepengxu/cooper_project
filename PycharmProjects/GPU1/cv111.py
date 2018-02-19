# -*- coding: UTF-8 -*-
#读取某个文件夹下的所有图片并生成一个[epoch_size   ,  h*w]的矩阵
import tensorflow as tf
import numpy as np
import PIL.Image as Image
import time
import cv2 as cv
import os
dir_path='/home/cooper/图片/image_209'
file_list=os.listdir(dir_path)
n=file_list.__len__()#样本个数
result_example=np.array([[]])
time_reslut_onehot=np.array([[]])
NUM_CLASSES=10
#print "result",result.shape
i=1
for name in file_list:
#    int(name[0])
    image=Image.open(dir_path+'/'+name).convert('L')
#    image=np.expand_dims(image)
    image_arr=np.array(image).reshape(2073600)
    image_arr=np.expand_dims(image_arr,0)
#    print image_arr.shape
    if i==1:
        time_reslut_onehot=np.array([[int(name[0])]])
        result_example=image_arr
        i=0
    else:
        time_reslut_onehot=np.row_stack((time_reslut_onehot,name[0]))
        result_example=np.row_stack((result_example,image_arr))
print result_example.shape
#the size of image 2073600
indices=tf.expand_dims(tf.range(0,n,),1)
concated=tf.concat([indices,time_reslut_onehot],1)
onehot_labels=tf.sparse_to_dense(concated,[n,NUM_CLASSES],1.0,0.0)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    onehot=sess.run(onehot_labels)
    print onehot
def read_picture(dir_path,NUM_CLASSES,image_h,image_w):
    '''
    this function will return two parmater:
        result_example:[epoch_size,image_h*image_w]
        onehot [epoch,num_classes]
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
    NUM_CLASSES = 10
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
        return result_example,onehot