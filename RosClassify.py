import numpy as np
import tensorflow as tf
import argparse
import os
import time
import pickle

import to_blocks_working
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
	parser.add_argument('--batch_size', type=int, default=8, help='size of mini batch')
        parser.add_argument('--model', default='my', help='Model name [default: my]')
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
        model_dir = 'saved_models'
        with open(os.path.join(model_dir, 'config.pkl'), 'rb') as f:
                saved_args = pickle.load(f)

        self.graph1 = tf.Graph()
        with self.graph1.as_default():
            self.block_loader, self.block_in =  to_blocks_working.init_block_model(2048, block_size=10.0, stride=10.0, num_in=5)
            # Create a PointNet model with the saved arguments
            saved_args.batch_size=args.batch_size
            saved_args.model=args.model
            self.model = Model(saved_args)
            self.sess=tf.InteractiveSession()
            # Add all the variables to the list of variables to be saved
            saver = tf.train.Saver(tf.global_variables())
            # restore the model
            saver.restore(self.sess, os.path.join('log', 'model.ckpt'))

            self.pred=self.model.pred
            nclasses=self.pred.shape[-1]
            print nclasses
            self.pred=tf.reshape(self.pred, (-1,nclasses))
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
        data_out = pc2.read_points(msg, skip_nans=True)
        a=np.array(list(data_out))
        a=a[:,:4]
        a[:,3]=a[:,3]/255
        b = np.expand_dims(np.arange(a.shape[0]),1)
        data=np.concatenate((a,b),axis=1)
        start = time.time()
        max_points=a.shape[0]
        #print a.shape
        #blocks
        feed = {self.block_in:data}
        start_time=time.time()
        data = np.array(self.sess.run([self.block_loader], feed)[0])
        indexes=data[:,:,7]
        x   =data[:,:,:7]
        #data[:,:,3]=(data[:,:,3]+2000)/4000
        mid_time=time.time()
        
        #pointnet
        feed = {self.model.input_data: x, self.model.is_training: False}
        F = self.sess.run(self.pred, feed)
        indexes=np.reshape(indexes,-1)
        result = self.sess.run(self.InSum.out_points, {self.InSum.input:F, self.InSum.indices:indexes, self.InSum.points:a})
        end_time=time.time()
        print mid_time-start_time, end_time-mid_time, end_time-start_time
        self.PublishCloud(result)

        print( result.shape)
        end = time.time()
    def main(self):
        rospy.spin()


if __name__ == '__main__':
	main()
