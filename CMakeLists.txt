cmake_minimum_required(VERSION 3.0.0)
project(zeroDL VERSION 0.1.0)

include(CTest)
enable_testing()

set(CMAKE_CXX_STANDARD 11)

add_executable(zeroDL main.cpp)

# 本当はtarget_include_directoriesで個別に指定した方が良い
include_directories(/usr/include/eigen3)
include_directories(/usr/include/python3.6)
include_directories(include)
include_directories(datasets/include)
include_directories(ch5/include)

add_subdirectory(src)
add_subdirectory(datasets/src)
add_subdirectory(test)
add_subdirectory(ch5)
add_subdirectory(ch5/src)
add_subdirectory(ch6)

# link_directories(include)
# link_directories(datasets/include)
# link_directories(ch5/include)

# target_link_libraries(zeroDL include)
# target_link_libraries(zeroDL datasets/include)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

# 変数の表示
# get_cmake_property(_variableNames CACHE_VARIABLES)
# list (SORT _variableNames)
# foreach (_variableName ${_variableNames})
#     message(STATUS "${_variableName}=${${_variableName}}")
# endforeach()
