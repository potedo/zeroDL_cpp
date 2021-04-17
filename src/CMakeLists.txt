cmake_minimum_required(VERSION 3.0.0)
add_library(zeroDLcpp STATIC
    activation.cpp
    layer.cpp
    optimizer.cpp
    sample_network.cpp
    loss.cpp
    numerical_gradient.cpp
)

target_include_directories(zeroDLcpp
    PUBLIC /mnt/c/Users/lank4/Desktop/zeroDL_cpp/include
)