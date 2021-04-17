cmake_minimum_required(VERSION 3.0.0)

add_executable(apple_cal_graph apple_cal_graph.cpp)
add_executable(affine affine.cpp)
add_executable(mul_layer mul_layer.cpp)
add_executable(relu relu.cpp)
add_executable(sigmoid sigmoid.cpp)
add_executable(softmax_with_loss softmax_with_loss.cpp)

# link_directories(ch5/include)

target_link_libraries(apple_cal_graph simple_dnn_lib)
target_link_libraries(affine simple_dnn_lib)
target_link_libraries(mul_layer simple_dnn_lib)
target_link_libraries(relu simple_dnn_lib)
target_link_libraries(sigmoid simple_dnn_lib)
target_link_libraries(softmax_with_loss simple_dnn_lib)