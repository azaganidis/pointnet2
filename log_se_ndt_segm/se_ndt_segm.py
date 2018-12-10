import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module

def placeholder_inputs(batch_size, num_point, point_size):
    pointclouds_pl = tf.placeholder(tf.float32, [batch_size, num_point, point_size])
    labels_pl = tf.placeholder(tf.int32, [batch_size, num_point])
    return pointclouds_pl, labels_pl


def RandomRotation(data, is_training):
    angle=tf.random_uniform([3], minval=-1, maxval=1)
    c=tf.cos(angle)
    s=tf.sin(angle)
    R=tf.stack([ tf.stack([c[1]*c[2], -c[1]*s[2], s[1], 0]),
        tf.stack([c[0]*s[2]+c[2]*s[0]*s[1], c[0]*c[2]-s[0]*s[1]*s[2], -c[1]*s[0], 0]),
        tf.stack([s[0]*s[2]-c[0]*c[2]*s[1], c[2]*s[0]+c[0]*s[1]*s[2], c[0]*c[1], 0]),
        [0,0,0,1]])
    R_e=tf.tile(tf.expand_dims(R,0),(data.get_shape()[0].value,1,1))
    data= tf.cond(is_training, lambda: tf.matmul(data,R_e), lambda: data)
    return data, R
def NoiseAugment(data,is_training):
    DS = tf.shape(data)
    noise=tf.random_uniform([DS[0],DS[1],3], minval=-0.001, maxval=0.001)
    ndata=tf.concat([data[...,:3]+noise , data[...,3:]], axis=-1)
    data=tf.cond(is_training, lambda:ndata, lambda:data)
    return data

#MAKE MOVE CLASS AUGMENT
def ClassAugment(point_cloud, labels, is_training):
    classes=tf.unique(tf.reshape(labels,[-1]))[0]
    augmented_class=classes[tf.random_uniform([1], minval=0, maxval=tf.shape(classes)[0]-1, dtype=tf.int32)[0]]
    rand_t=tf.random_uniform([2], minval=-0.10, maxval=0.10, dtype=tf.float32)
    rand_tz=tf.random_uniform([1], minval=-0.010, maxval=0.010, dtype=tf.float32)
    rand_t=tf.concat([rand_t,rand_tz, tf.zeros([point_cloud.shape[-1]-3])], 0)
    rand_t=tf.reshape(rand_t, (1,1,-1))
    rand_t=tf.tile(rand_t, (point_cloud.shape[0], point_cloud.shape[1],1))
    augmented_class=tf.ones_like(labels)*augmented_class
    point_cloud_new=tf.where(augmented_class==labels, rand_t+point_cloud, point_cloud)
    return tf.cond(is_training, lambda: point_cloud_new, lambda: point_cloud)



def get_model(point_cloud, is_training, num_classes, bn_decay=None):
    """ Part segmentation PointNet, input is BxNx6 (XYZ NormalX NormalY NormalZ), output Bx50 """
    point_cloud, R=RandomRotation(point_cloud, is_training)
    point_cloud=NoiseAugment(point_cloud, is_training)
    #point_cloud=tf.Print(point_cloud, [R], summarize=16)

    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3])
    l0_points = tf.slice(point_cloud, [0,0,3], [-1,-1,point_cloud.shape[2]-3])
    #l0_xyz=tf.Print(l0_xyz, [tf.reduce_mean(l0_xyz,axis=1), tf.reduce_mean(l0_points,axis=1)])

    # Set Abstraction layers
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=1024, radius=0.02, nsample=64, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=256, radius=0.04, nsample=64, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=64, radius=0.04, nsample=8, mlp=[64,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer3')

    # Feature Propagation layers
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], is_training, bn_decay, scope='fa_layer1')
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay, scope='fa_layer2')
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, tf.concat([l0_xyz,l0_points],axis=-1), l1_points, [128,128,128], is_training, bn_decay, scope='fa_layer3')


    # FC layers
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    end_points['feats'] = net 
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.conv1d(net, num_classes, 1, padding='VALID', activation_fn=None, scope='fc2')

    return net, end_points


def get_loss(pred, label):
    """ pred: BxNxC,
        label: BxN, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,6))
        net, _ = get_model(inputs, tf.constant(True))
        print(net)
