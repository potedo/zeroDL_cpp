cmake_minimum_required(VERSION 3.0.0)

add_executable(eigen_sample_operation eigen_sample_operation.cpp)
add_executable(MLP_3layers MLP_3layers.cpp)
add_executable(sample_imshow sample_imshow.cpp)
add_executable(test_softmax_eigen test_softmax_eigen.cpp)
add_executable(vis_eigen_vector_revised vis_eigen_vector_revised.cpp)
add_executable(visualize_eigen_vector visualize_eigen_vector.cpp)
add_executable(visualize_relu visualize_relu.cpp)
add_executable(visualize_sigmoid visualize_sigmoid.cpp)
add_executable(visualize_step_function_sample visualize_step_function_sample.cpp)
add_executable(visualize_step_function visualize_step_function.cpp)

find_package(PythonLibs REQUIRED)

target_include_directories(sample_imshow PUBLIC ${PYTHON_INCLUDE_DIRS})
target_include_directories(vis_eigen_vector_revised PUBLIC ${PYTHON_INCLUDE_DIRS})
target_include_directories(visualize_eigen_vector PUBLIC ${PYTHON_INCLUDE_DIRS})
target_include_directories(visualize_relu PUBLIC ${PYTHON_INCLUDE_DIRS})
target_include_directories(visualize_sigmoid PUBLIC ${PYTHON_ICNLUDE_DIRS})
target_include_directories(visualize_step_function_sample PUBLIC ${PYTHON_INCLUDE_DIRS})
target_include_directories(visualize_step_function PUBLIC ${PYTHON_INCLUDE_DIRS})

target_link_libraries(MLP_3layers
    simple_dnn_lib
)

target_link_libraries(sample_imshow
    ${PYTHON_LIBRARIES}
)

target_link_libraries(test_softmax_eigen
    simple_dnn_lib
)

target_link_libraries(vis_eigen_vector_revised
    ${PYTHON_LIBRARIES}
)

target_link_libraries(visualize_eigen_vector
    ${PYTHON_LIBRARIES}
)

target_link_libraries(visualize_relu
    ${PYTHON_LIBRARIES}
    simple_dnn_lib
)

target_link_libraries(visualize_sigmoid
    ${PYTHON_LIBRARIES}
    simple_dnn_lib
)

target_link_libraries(visualize_step_function_sample
    ${PYTHON_LIBRARIES}
    simple_dnn_lib
)

target_link_libraries(visualize_step_function
    ${PYTHON_LIBRARIES}
    simple_dnn_lib
)