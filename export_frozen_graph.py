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
        model_dir = args.log_dir
        with open(os.path.join(model_dir, 'config.pkl'), 'rb') as f:
                saved_args = pickle.load(f)
        graph1 = tf.Graph()
        with graph1.as_default():
            # Create a PointNet model with the saved arguments
            saved_args.batch_size=args.batch_size
            saved_args.model=args.model
            model = Model(saved_args)
            sess=tf.InteractiveSession()
            # Add all the variables to the list of variables to be saved
            saver = tf.train.Saver(tf.global_variables())
            # restore the model
            saver.restore(sess, os.path.join(args.log_dir, 'model.ckpt'))
            # save frozen graph in file for use in C++ 
            tf.train.write_graph(model.freeze_session(sess, keep_var_names=["cloud_in"], output_names=["cloud_out", "layer3/points_out"]), 'deploy/build/models/', 'pointnet.pb', as_text=False)
            print "Frozen graph saved for deployment"

if __name__ == '__main__':
	main()
