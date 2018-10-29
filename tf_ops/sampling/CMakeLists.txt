cmake_minimum_required(VERSION 3.5)

set(CMAKE_C_COMPILER /opt/cuda/bin/gcc)
set(CMAKE_CXX_COMPILER /opt/cuda/bin/g++)

# get tensorflow include dirs, see https://www.tensorflow.org/how_tos/adding_an_op/
execute_process(COMMAND python3 -c "import tensorflow; print(tensorflow.sysconfig.get_include())" OUTPUT_VARIABLE Tensorflow_INCLUDE_DIRS)
execute_process(COMMAND python -c "import tensorflow; print(tensorflow.sysconfig.get_include())" OUTPUT_VARIABLE Tensorflow_INCLUDE_DIRS2)

find_package(CUDA)

# C++11 required for tensorflow
#set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${CMAKE_CXX_FLAGS}")
#set(CMAKE_CXX_FLAGS "-std=c++11 -fPIC -O2 ${CMAKE_CXX_FLAGS}")

# if GCC > 5
if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0 OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 5.0)
  set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0 ${CMAKE_CXX_FLAGS}")
endif()

#pass flags to c++ compiler
SET(CUDA_PROPAGATE_HOST_FLAGS ON)

set(SUBDIR
    "/home/anestis/.local/lib64/python2.7/site-packages/tensorflow/include/external/nsync/public")

# build the actual operation which can be used directory
include_directories(${Tensorflow_INCLUDE_DIRS}
    ${Tensorflow_INCLUDE_DIRS2}${SUBDIR}
    )
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY
    INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
      message(STATUS "dir='${dir}'")
  endforeach()

#create library
cuda_add_library(
    tf_sampling SHARED
    tf_sampling_g.cu
    tf_sampling.cpp)
    
    #OPTIONS -gencode=arch=compute_20,code=sm_20)

#add_library(dense SHARED dense.cc)
