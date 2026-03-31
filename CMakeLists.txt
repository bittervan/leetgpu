cmake_minimum_required(VERSION 3.18)

project(leetgpu LANGUAGES CXX CUDA)

option(LEETGPU_RUN_CHECK_ON_BUILD "Run compare targets during the default build" OFF)
option(LEETGPU_EXPORT_COMPILE_COMMANDS "Export compile_commands.json for supported generators" ON)

if(LEETGPU_EXPORT_COMPILE_COMMANDS)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

enable_testing()

add_subdirectory(matrix_multiplication)

if(TARGET matrix_multiplication_check)
    add_custom_target(check DEPENDS matrix_multiplication_check)
endif()
