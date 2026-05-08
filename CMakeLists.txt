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

add_subdirectory(convolution_1d)
add_subdirectory(image_invert)
add_subdirectory(matrix_multiplication)
add_subdirectory(reduction)
add_subdirectory(softmax)
add_subdirectory(vector_add)
add_subdirectory(matrix_transpose)

add_custom_target(check)

if(TARGET convolution_1d_check)
    add_dependencies(check convolution_1d_check)
endif()

if(TARGET image_invert_check)
    add_dependencies(check image_invert_check)
endif()

if(TARGET matrix_multiplication_check)
    add_dependencies(check matrix_multiplication_check)
endif()

if(TARGET reduction_check)
    add_dependencies(check reduction_check)
endif()

if(TARGET softmax_check)
    add_dependencies(check softmax_check)
endif()

if(TARGET vector_add_check)
    add_dependencies(check vector_add_check)
endif()

if(TARGET matrix_transpose_check)
    add_dependencies(check matrix_transpose_check)
endif()
