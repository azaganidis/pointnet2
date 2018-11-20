#/bin/bash
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )

g++ -std=c++11 -shared tf_interpolate.cpp -o tf_interpolate_so.so -I/home/anestis/.local/lib64/python2.7/site-packages/tensorflow/include/external/nsync/public/ -I/opt/cuda/include -fPIC -L/opt/cuda/lib64 -lcudart ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -ltensorflow_framework
