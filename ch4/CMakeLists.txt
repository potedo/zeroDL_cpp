cmake_minimum_required(VERSION 3.0.0)

add_executable(batch_cross_entropy batch_cross_entropy.cpp)
add_executable(check_eigen_argmax check_eigen_argmax.cpp)
add_executable(check_eigen_index check_eigen_index.cpp)
add_executable(cross_entropy cross_entropy.cpp)
add_executable(differential differential.cpp)
add_executable(gradient gradient.cpp)
add_executable(gradient_descent gradient_descent.cpp)
add_executable(mean_squared_error mean_squared_error.cpp)
add_executable(mnist_two_layer_net mnist_two_layer_net.cpp)
add_executable(partial_diff partial_diff.cpp)
add_executable(simpleNet simpleNet.cpp)
add_executable(test_two_layer_net test_two_layer_net.cpp)
add_executable(train_mnist_two_layer_net train_mnist_two_layer_net.cpp)

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

target_link_libraries(differential
    ${PYTHON_LIBRARIES}
)

target_link_libraries(gradient
    ${PYTHON_LIBRARIES}
)

target_link_libraries(mnist_two_layer_net
    simple_dnn_lib
    mnist
)

target_link_libraries(partial_diff
    ${PYTHON_LIBRARIES}
)

target_link_libraries(simpleNet
    simple_dnn_lib
)

target_link_libraries(test_two_layer_net
    simple_dnn_lib
)

target_link_libraries(train_mnist_two_layer_net
    ${PYTHON_LIBRARIES}
    simple_dnn_lib
    mnist
)