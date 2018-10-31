#/bin/bash
/opt/cuda/bin/nvcc tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# TF1.2
#g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /opt/cuda/include -lcudart -L /opt/cuda/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# TF1.4
g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I /home/anestis/.local/lib64/python2.7/site-packages/tensorflow/include -I /opt/cuda/include -I /home/anestis/.local/lib64/python2.7/site-packages/tensorflow/include/external/nsync/public -lcudart -L /opt/cuda/lib64/ -L/home/anestis/.local/lib64/python2.7/site-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
