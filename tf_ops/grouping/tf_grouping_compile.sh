#/bin/bash
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )

/opt/cuda/bin/nvcc -std=c++11 -c -o tf_grouping_g.cu.o tf_grouping_g.cu -I/home/anestis/.local/lib64/python2.7/site-packages/tensorflow/include -D_GLIBCXX_USE_CXX11_ABI=1 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 -shared tf_grouping_g.cu.o tf_grouping.cpp -o tf_grouping_so.so -I/home/anestis/.local/lib64/python2.7/site-packages/tensorflow/include/external/nsync/public/ -I/opt/cuda/include -fPIC -L/opt/cuda/lib64 -lcudart ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -ltensorflow_framework

