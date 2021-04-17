cmake_minimum_required(VERSION 3.0.0)

add_library(ch5_library STATIC
    simple_layer.cpp
    simple_activation.cpp
    simple_loss.cpp
    simple_mlp.cpp
    two_layer_net.cpp
    numerical_gradient.cpp
)

target_include_directories(ch5_library
    PUBLIC /mnt/c/Users/lank4/Desktop/zeroDL_cpp/ch5/include
)
