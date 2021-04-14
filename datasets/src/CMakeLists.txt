cmake_minimum_required(VERSION 3.0.0)
add_library(mnist STATIC mnist.cpp)

target_include_directories(zeroDL
    PUBLIC /mnt/c/Users/lank4/Desktop/zeroDL_cpp/datasets/include
)