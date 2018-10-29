import os
import sys
import importlib
import tensorflow as tf
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))
class Pointnet2Interface:
    def __init__(self, args):
        self.batch_size=args.batch_size
        MODEL_NAME=args.model
        model = importlib.import_module(MODEL_NAME) # import network module
        self.is_training= tf.placeholder(tf.bool, shape=())
        self.input_data,labels_pl = model.placeholder_inputs(args.batch_size, args.input_dim[0], args.input_dim[1])
        self.pred, self.end_points= model.get_model(self.input_data, self.is_training, args.output_dim)
