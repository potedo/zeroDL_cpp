cmake_minimum_required(VERSION 3.0.0)
add_library(zeroDLcpp STATIC
    activation.cpp
    layer.cpp
    optimizer.cpp
    sample_network.cpp
    loss.cpp
    numerical_gradient.cpp
)
# add_library(activation STATIC activation.cpp)
# add_library(layer STATIC layer.cpp)
# add_library(optimizer STATIC optimizer.cpp)
# add_library(sample_network STATIC sample_network.cpp)
# add_library(loss STATIC loss.cpp)
# add_library(numerical_gradient STATIC numerical_gradient.cpp)

target_include_directories(zeroDL
    PUBLIC /mnt/c/Users/lank4/Documents/cpp_practice/zeroDL_re_impl/include
)