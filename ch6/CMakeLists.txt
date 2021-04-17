cmake_minimum_required(VERSION 3.0.0)

add_executable(Adam Adam.cpp)
add_executable(AdaGrad AdaGrad.cpp)
add_executable(RMSprop RMSprop.cpp)
add_executable(SGD-Momentum SGD-Momentum.cpp)

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

target_link_libraries(Adam
    zeroDLcpp
    ${PYTHON_LIBRARIES}
)

target_link_libraries(AdaGrad
    zeroDLcpp
    ${PYTHON_LIBRARIES}
)

target_link_libraries(RMSprop
    zeroDLcpp
    ${PYTHON_LIBRARIES}
)

target_link_libraries(SGD-Momentum
    zeroDLcpp
    ${PYTHON_LIBRARIES}
)