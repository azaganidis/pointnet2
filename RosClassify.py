#!/usr/bin/python
import numpy as np
import tensorflow as tf
import argparse
import os
import time
import pickle
import resample_cloud
from itertools import chain

#import to_blocks_working
from interface_pointnet2 import Pointnet2Interface as Model
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
import time
class IndexSum():
    def __init__(self, point_len, feature_len):
        self.input = tf.placeholder(tf.float32, shape=(None, feature_len))
        self.points = tf.placeholder(tf.float32, shape=(None, point_len))
        self.indices = tf.placeholder(tf.int32, shape=(None))
        #number = tf.shape(tf.unique(tf.contrib.framework.sort(self.indices)).y)
        ordering_of_indices = tf.contrib.framework.argsort(self.indices)
        sorted_by_index = tf.gather(self.input, ordering_of_indices)
        #sorted_indexes = tf.gather(self.indices, ordering_of_indices)
        sorted_indexes = tf.contrib.framework.sort(self.indices)
        sum_of_values= tf.segment_sum(sorted_by_index, sorted_indexes)
        clusters = tf.argmax(sum_of_values, axis=1)
        unique_indices = tf.unique(sorted_indexes).y

        clusters = tf.gather(clusters,unique_indices) 
        out_points = tf.gather(self.points[:,:-1], unique_indices)

        self.out_points=tf.concat([out_points, tf.expand_dims(tf.cast(clusters, tf.float32),1)], axis=1)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', type=int, default=1, help='size of mini batch')
        parser.add_argument('--model', default='my', help='Model name [default: my]')
        parser.add_argument('--log_dir', default='log', help='Log directory [default: log]')
	#parser.add_argument('--input_dim', type=int, default=[2048, 7, 1],
	#                    help='dim of input')
	#parser.add_argument('--maxn', type=int, default=2048,
	#                    help='max number of point cloud size')
	args = parser.parse_args()
        rospy.init_node('semantics', anonymous=True)
	Semantics = RosSemantics(args)
        Semantics.main()


class RosSemantics():
    def __init__(self, args):
        # load saved model
        model_dir = args.log_dir
        with open(os.path.join(model_dir, 'config.pkl'), 'rb') as f:
                saved_args = pickle.load(f)

        self.graph1 = tf.Graph()
        with self.graph1.as_default():
            #self.block_loader, self.block_in =  to_blocks_working.init_block_model(2048, block_size=10.0, stride=10.0, num_in=5)
            # Create a PointNet model with the saved arguments
            saved_args.batch_size=args.batch_size
            saved_args.model=args.model
            self.model = Model(saved_args)
            self.sess=tf.InteractiveSession()
            # Add all the variables to the list of variables to be saved
            saver = tf.train.Saver(tf.global_variables())
            # restore the model
            saver.restore(self.sess, os.path.join(args.log_dir, 'model.ckpt'))
            # save frozen graph in file for use in C++ 
            # tf.train.write_graph(self.model.freeze_session(self.sess, keep_var_names=["cloud_in"], output_names=["cloud_out", "layer3/points_out"]), 'deploy/build/models/', 'pointnet.pb', as_text=False)
            # print "Frozen graph saved for deployment"

            self.pred=self.model.pred
            nclasses=self.pred.shape[-1]
            self.InSum = IndexSum(4,nclasses)
        rospy.Subscriber("/velodyne_points", PointCloud2, self.callback)
        self.publisher  = rospy.Publisher('semantic', PointCloud2, queue_size=10)

    def PublishCloud(self, data):
        fields=[ pc2.PointField('x',0,pc2.PointField.FLOAT32,1),
                pc2.PointField('y',4,pc2.PointField.FLOAT32,1),
                pc2.PointField('z',8,pc2.PointField.FLOAT32,1),
                pc2.PointField('intensity',12,pc2.PointField.FLOAT32,1)]
        a= PointCloud2
        header=Header()
        header.stamp = rospy.Time.now()
        header.frame_id="velodyne"
        msP = pc2.create_cloud(header,fields, data)  
        self.publisher.publish(msP)

    def callback(self, msg):
        t1 = time.time()
        data_out = pc2.read_points(msg, skip_nans=True)
        t2=time.time()
        data=np.reshape(np.fromiter(chain.from_iterable(data_out), float), (-1,4))
        t3=time.time()
        feed = {self.model.input_data: data, self.model.is_training: False}
        result = self.sess.run(self.pred, feed)
        t4=time.time()
        self.PublishCloud(result)
        t5=time.time()
        print t2-t1, t3-t2, t4-t3, t5-t4, t5-t1
    def main(self):
        rospy.spin()


if __name__ == '__main__':
	main()
