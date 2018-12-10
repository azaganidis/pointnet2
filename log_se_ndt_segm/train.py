'''
    Single-GPU training.
    Will use H5 dataset in default. If using normal, will shift to the normal dataset.
'''
import argparse
import math
from datetime import datetime
import time
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
import socket
import importlib
import os
import sys
import pickle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import tf_util
#import modelnet_dataset
#import modelnet_h5_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='my', help='Model name [default: my]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--input_dim', type=int, default=[15000, 4, 1], help='dim of input')
parser.add_argument('--output_dim', type=int, default=8, help='dim of output')
parser.add_argument('--max_epoch', type=int, default=251, help='Epoch to run [default: 251]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=20000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--normal', action='store_true', help='Whether to use normal information')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

POINT_SIZE=FLAGS.input_dim[1]
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.input_dim[0]
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

NUM_CLASSES = FLAGS.output_dim
with open(os.path.join(LOG_DIR, 'config.pkl'), 'wb') as f:
        pickle.dump(FLAGS, f)

def _parse_function(example_proto):
    keys_to_features={ 'data': tf.VarLenFeature(tf.float32), 'label': tf.VarLenFeature(tf.int64) }
    parsed_features=tf.parse_single_example(example_proto,keys_to_features)
    data=tf.sparse_tensor_to_dense(parsed_features['data'],default_value=0)
    data=tf.reshape(data,(NUM_POINT,POINT_SIZE))
    label=tf.sparse_tensor_to_dense(parsed_features['label'],default_value=0)
    label=tf.reshape(label,(NUM_POINT,1))
    label= tf.squeeze(label)
    label=tf.cast(label, dtype=tf.float32)
    return data,label
data_dir = '/home/anestis/disk/unsupervised/tfrecords/'

# Shapenet official train/test split
'''
if FLAGS.normal:
    assert(NUM_POINT<=10000)
    DATA_PATH = os.path.join(ROOT_DIR, 'data/modelnet40_normal_resampled')
    TRAIN_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='train', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
    TEST_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='test', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
else:
    assert(NUM_POINT<=2048)
    TRAIN_DATASET = modelnet_h5_dataset.ModelNetH5Dataset(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'), batch_size=BATCH_SIZE, npoints=NUM_POINT, shuffle=True)
    TEST_DATASET = modelnet_h5_dataset.ModelNetH5Dataset(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'), batch_size=BATCH_SIZE, npoints=NUM_POINT, shuffle=False)
'''

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
        #if True:
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, POINT_SIZE)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter
            # for you every time it trains.
            batch = tf.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pointclouds_pl=MODEL.ClassAugment(pointclouds_pl, labels_pl,is_training_pl)
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, NUM_CLASSES, bn_decay=bn_decay)
            MODEL.get_loss(pred, labels_pl)
            losses = tf.get_collection('losses')
            total_loss = tf.add_n(losses, name='total_loss')
            tf.summary.scalar('total_loss', total_loss)
            for l in losses + [total_loss]:
                tf.summary.scalar(l.op.name, l)

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            correct_sum = tf.reduce_sum(tf.cast(correct, tf.float32)) 
            accuracy = correct_sum/ float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)

            print "--- Get training operator"
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(total_loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        train_data_path = data_dir+'train.tfrecords.noblock'  # address to save the hdf5 file
        valid_data_path = data_dir+'valid.tfrecords.noblock'  # address to save the hdf5 file
        sess = tf.Session(config=config)

        #Train set
        dataset_training = tf.data.TFRecordDataset([train_data_path])
        dataset_training = dataset_training.map(_parse_function)
        dataset_training = dataset_training.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
        dataset_training = dataset_training.batch(BATCH_SIZE, drop_remainder=True)
        train_iterator   = dataset_training.make_initializable_iterator()
        t_i_next=train_iterator.get_next()
        #validation set
        dataset_validation = tf.data.TFRecordDataset([valid_data_path])
        dataset_validation = dataset_validation.map(_parse_function)
        validation_iterator   = dataset_validation.batch(BATCH_SIZE, drop_remainder=True).make_initializable_iterator()
        v_i_next=validation_iterator.get_next()
        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        if os.path.exists(os.path.join(LOG_DIR, "model.ckpt.index")):
            saver.restore(sess,os.path.join(LOG_DIR, "model.ckpt"))
        else:
            init = tf.global_variables_initializer()
            sess.run(init)

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': total_loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'train_iterator':train_iterator,
               'validation_iterator':validation_iterator,
               't_i_next':t_i_next,
               'v_i_next':v_i_next,
               'correct_sum':correct_sum,
               'end_points': end_points}

        best_acc = -1
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer)

            # Save the variables to disk.
            if epoch % 1 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    log_string(str(datetime.now()))

    sess.run(ops['train_iterator'].initializer)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    start=time.time()
    try:
        while True:
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            cur_batch_data,cur_batch_label= sess.run(ops['t_i_next'])

            feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                         ops['labels_pl']: cur_batch_label,
                         ops['is_training_pl']: is_training,}
            summary, step, _, loss_val, correct = sess.run([ops['merged'], ops['step'],
                ops['train_op'], ops['loss'], ops['correct_sum']], feed_dict=feed_dict)#, options=options, run_metadata=run_metadata)
            #fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            #chrome_trace = fetched_timeline.generate_chrome_trace_format()
            #with open('timeline_01.json', 'w') as f:
            #    f.write(chrome_trace)
            #summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            #    ops['loss'], ops['pred']], feed_dict=feed_dict)
            train_writer.add_summary(summary, step)
            #pred_val = np.argmax(pred_val, 2)
            #correct = np.sum(pred_val == cur_batch_label)
            total_correct += correct
            total_seen += BATCH_SIZE*NUM_POINT
            loss_sum += loss_val
            PRINT_EVERY=int(50/BATCH_SIZE)
            if (batch_idx+1)%PRINT_EVERY == 0:
                print( "training {}/{} , train_loss = {:.6f}, accuracy = {:.6f}, time = {:.6f}" .format( batch_idx*BATCH_SIZE+1, 1500, loss_sum/PRINT_EVERY, (total_correct/float(total_seen)), (time.time()-start)/PRINT_EVERY/BATCH_SIZE))
                start=time.time()
                #log_string(' ---- batch: %03d ----' % (batch_idx+1))
                #log_string('mean loss: %f' % (loss_sum / 50))
                #log_string('accuracy: %f' % (total_correct / float(total_seen)))
                total_correct = 0
                total_seen = 0
                loss_sum = 0
            batch_idx += 1
            #if batch_idx==1:
            #    last_cloud=np.concatenate((cur_batch_data[-1,...], np.expand_dims(pred_val[-1,...], 1)), axis=1)
            #    np.savetxt('cloud_out/c'+str(batch_idx)+'.txt', last_cloud)
            #    last_cloud=np.concatenate((cur_batch_data[-1,...], np.expand_dims(cur_batch_label[-1,...], 1)), axis=1)
            #    np.savetxt('cloud_out/C'+str(batch_idx)+'.txt', last_cloud)
    except tf.errors.OutOfRangeError:
        return

def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    shape_ious = []
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    
    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))
    sess.run(ops['validation_iterator'].initializer)
    
    more_samples=True
    while more_samples:
        try:
            cur_batch_data,cur_batch_label= sess.run(ops['v_i_next'])
        except tf.errors.OutOfRangeError:
            more_samples=False
            break
        finally:
            feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                         ops['labels_pl']: cur_batch_label,
                         ops['is_training_pl']: is_training}
            summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['loss'], ops['pred']], feed_dict=feed_dict)
            test_writer.add_summary(summary, step)
            pred_val = np.argmax(pred_val, 2)
            correct = np.sum(pred_val == cur_batch_label)
            total_correct += correct
            total_seen +=BATCH_SIZE*NUM_POINT
            loss_sum += loss_val
            batch_idx += 1
            CL = np.reshape(cur_batch_label, (-1,)).astype(int)
            PV = np.reshape(pred_val, (-1,)).astype(int)
            for i in range(0, BATCH_SIZE*NUM_POINT):
                l = CL[i]
                total_seen_class[l] += 1
                total_correct_class[l] += (PV[i] == l)
            if batch_idx==1:
                last_cloud=np.concatenate((cur_batch_data[-1,...], np.expand_dims(pred_val[-1,...], 1)), axis=1)
                np.savetxt('cloud_out/ct'+str(batch_idx)+'.txt', last_cloud)
                last_cloud=np.concatenate((cur_batch_data[-1,...], np.expand_dims(cur_batch_label[-1,...], 1)), axis=1)
                np.savetxt('cloud_out/Ct'+str(batch_idx)+'.txt', last_cloud)
    
    log_string('eval mean loss: %f' % (loss_sum / float(batch_idx)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
    print (np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))
    EPOCH_CNT += 1

    return total_correct/float(total_seen)


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
