import os
import numpy as np
import sys
import tensorflow as tf
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from tqdm import tqdm
import resample_cloud

def _float_array_feature(array):
    return tf.train.Feature(float_list=tf.train.FloatList(value=array))
def _int_array_feature(array):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=array))
# Constants
data_dtype = 'float32'
label_dtype = 'uint8'

data_dir_in = '/home/anestis/disk/unsupervised/train'
data_dir_out = '/home/anestis/disk/unsupervised'
file_extension = '.bin'

data_label_files = []
scene_files = []

for root, dirs, files in os.walk(data_dir_in):
	for file in files:
		if file.endswith(file_extension):
			print(os.path.join(root, file))
			data_label_files.append(os.path.join(root, file))
			scene_files.append(root)

print "in total " + str(len(data_label_files)) + " point cloud files."

# Set paths
#h5_filelist = os.path.join(BASE_DIR, 'meta/all_scan_label.txt')
#data_label_files = [os.path.join(data_dir, line.rstrip()) for line in open(filelist)]
output_dir = os.path.join(data_dir_out, 'tfrecords')
if not os.path.exists(output_dir):
	os.mkdir(output_dir)
train_fname= os.path.join(data_dir_out, 'train.tfrecords.noblock')
valid_fname= os.path.join(data_dir_out, 'valid.tfrecords.noblock')
print('Writing',train_fname)
print('Writing',valid_fname)

# --------------------------------------
# ----- BATCH WRITE TO TFRECORD -----
# --------------------------------------
data_orig = tf.placeholder(tf.float32,shape=(None,5))
data_resample= resample_cloud.init_resample2(data_orig, 15000, num_in=5)
sess = tf.InteractiveSession()

sample_cnt_train = 0
sample_cnt_valid = 0
with tf.python_io.TFRecordWriter(train_fname) as t_writer, tf.python_io.TFRecordWriter(valid_fname) as v_writer:
    #valid_scene = ['sg284', 'sg272']
    valid_scene = ['sg272']
    train_scene = ['sg271','sg274','sg275','sg279','neugasse1', 'untermaederbrunnen1', 'untermaederbrunnen3'] 
    #valid_scene = ['bildstein1','bildstein3','domfountain1','neugasse1','sg271','sg274','sg284', 'untermaederbrunnen1']
    #train_scene = ['sg272','sg275','sg279','domfountain3','bildstein5', 'untermaederbrunnen3', 'domfountain2', ] 
    for i, data_label_filename in enumerate(tqdm(data_label_files)):
        scene_=scene_files[i].split('/')[-1].split('_')
        scene_simple = scene_[0]+scene_[1][-1]
        if scene_simple not in (train_scene+valid_scene):
            continue
        data_raw = np.fromfile(data_label_filename,dtype=np.float32)
        data_raw = data_raw.reshape((-1,5))
        data_raw[:,3]=(data_raw[:,3]+2035)/4071
        data_raw[:,4]=data_raw[:,4]-1
        data = sess.run(data_resample, {data_orig:data_raw})
        data=np.array(data)
        label=data[:,4]
        data=data[:,:4]
        print scene_simple

        data=data.astype(np.float).reshape(-1)
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'data': _float_array_feature(data.flatten()),
                    'label': _int_array_feature(label.astype(int).flatten()),
                }
            )
        )
        if scene_simple in train_scene:
            sample_cnt_train += 1
            t_writer.write(example.SerializeToString())
        if scene_simple in valid_scene:
            sample_cnt_valid += 1
            v_writer.write(example.SerializeToString())

print("Total training samples: {0}".format(sample_cnt_train))
print("Total validation samples: {0}".format(sample_cnt_valid))
