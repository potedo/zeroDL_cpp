cmake_minimum_required(VERSION 3.0.0)

add_executable(train_convnet train_convnet.cpp)

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

target_link_libraries(train_convnet
    zeroDLcpp
    mnist
)