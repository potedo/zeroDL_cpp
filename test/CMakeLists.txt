cmake_minimum_required(VERSION 3.0.0)

add_executable(suddle_point suddle_point.cpp)
add_executable(test_layer test_layer.cpp)
add_executable(test_batchnorm test_batchnorm.cpp)
add_executable(test_multi_layer_net test_multi_layer_net.cpp)
add_executable(train_multi_layer_net train_multi_layer_net.cpp)
add_executable(train_two_layer_mlp train_two_layer_mlp.cpp)
add_executable(test_trainer test_trainer.cpp)

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

target_link_libraries(suddle_point
    zeroDLcpp
    ${PYTHON_LIBRARIES}
)

target_link_libraries(test_layer
    zeroDLcpp
)

target_link_libraries(test_batchnorm
    zeroDLcpp
)

target_link_libraries(test_multi_layer_net
    zeroDLcpp
)

target_link_libraries(train_multi_layer_net
    zeroDLcpp
    mnist
    ${PYTHON_LIBRARIES}
)

target_link_libraries(train_two_layer_mlp
    zeroDLcpp
    mnist
    ${PYTHON_LIBRARIES}
)

target_link_libraries(test_trainer
    zeroDLcpp
    mnist
)