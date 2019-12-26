cmake_minimum_required(VERSION 3.5)
project(celerity-std LANGUAGES CXX)

set(TARGET celerity-std CMAKE_CXX_STANDARD 17)
set(TARGET celerity-std CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(hipSYCL REQUIRED)
find_package(Celerity REQUIRED)

add_subdirectory(examples/basic)
add_subdirectory(examples/simple)
#add_subdirectory(examples/simple_actions)
add_subdirectory(examples/matmul)
#add_subdirectory(examples/wave_sim)
#add_subdirectory(examples/convolution)
