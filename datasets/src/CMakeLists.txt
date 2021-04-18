cmake_minimum_required(VERSION 3.0.0)
add_library(mnist STATIC mnist.cpp)

# 指定しているのは datasets/include であることに注意
target_include_directories(zeroDL
    PUBLIC ../include
)