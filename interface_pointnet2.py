import os
import sys
import importlib
import tensorflow as tf
import numpy as np
import resample_cloud
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))
class Pointnet2Interface:
    def __init__(self, args):
        scale=(130.0, 1)
        self.batch_size=args.batch_size
        MODEL_NAME=args.model
        model = importlib.import_module(MODEL_NAME) # import network module
        self.is_training= tf.placeholder_with_default(False, shape=(), name="is_training")
        self.input_data=tf.placeholder(tf.float32, shape=( None, args.input_dim[1]) , name="cloud_in")
        data_out = resample_cloud.init_resample2(self.input_data, args.input_dim[0], 4, scale=scale)
        data=tf.expand_dims(data_out, 0)
        pred, self.end_points= model.get_model(data, self.is_training, args.output_dim)
        pred=tf.cast(tf.argmax(pred, axis=2), tf.float32)
        pred=tf.squeeze(pred, 0)
        self.pred=tf.concat((data_out[:,:3]*scale[0], tf.expand_dims(pred,1)), 1, name="cloud_out")
    def freeze_session(self, session, keep_var_names=None, output_names=None, clear_devices=True):
        """
        Freezes the state of a session into a pruned computation graph.

        Creates a new computation graph where variable nodes are replaced by
        constants taking their current value in the session. The new graph will be
        pruned so subgraphs that are not necessary to compute the requested
        outputs are removed.
        @param session The TensorFlow session to be frozen.
        @param keep_var_names A list of variable names that should not be frozen,
                              or None to freeze all the variables in the graph.
        @param output_names Names of the relevant graph outputs.
        @param clear_devices Remove the device directives from the graph for better portability.
        @return The frozen graph definition.
        """
        graph = session.graph
        with graph.as_default():
            freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
            output_names = output_names or []
            output_names += [v.op.name for v in tf.global_variables()]
            input_graph_def = graph.as_graph_def()
            if clear_devices:
                for node in input_graph_def.node:
                    node.device = ""
            frozen_graph = tf.graph_util.convert_variables_to_constants(
                session, input_graph_def, output_names, freeze_var_names)
            return frozen_graph
