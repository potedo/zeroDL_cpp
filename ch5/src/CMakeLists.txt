cmake_minimum_required(VERSION 3.0.0)

add_library(simple_dnn_lib STATIC
    simple_layer.cpp
    simple_activation.cpp
    simple_loss.cpp
    simple_mlp.cpp
    two_layer_net.cpp
    numerical_gradient.cpp
)

target_include_directories(simple_dnn_lib
    PUBLIC /mnt/c/Users/lank4/Desktop/zeroDL_cpp/ch5/include
)
