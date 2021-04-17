cmake_minimum_required(VERSION 3.0.0)

add_executable(check_eigen check_eigen.cpp)
add_executable(and_net and_net.cpp)
add_executable(nand_net nand_net.cpp)
add_executable(nor_net nor_net.cpp)
add_executable(or_net or_net.cpp)
add_executable(xor_net xor_net.cpp)

target_link_libraries(and_net
    simple_dnn_lib
)

target_link_libraries(nand_net
    simple_dnn_lib
)

target_link_libraries(nor_net
    simple_dnn_lib
)

target_link_libraries(or_net
    simple_dnn_lib
)

target_link_libraries(xor_net
    simple_dnn_lib
)