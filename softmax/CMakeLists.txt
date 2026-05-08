find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(SOFTMAX_TARGET softmax_runner)
set(SOFTMAX_TESTDATA_DIR ${CMAKE_CURRENT_BINARY_DIR}/testdata)
set(SOFTMAX_TESTDATA_STAMP ${SOFTMAX_TESTDATA_DIR}/.generated.stamp)
set(SOFTMAX_TESTDATA_FILES
    ${SOFTMAX_TESTDATA_DIR}/case_00_example_3.txt
    ${SOFTMAX_TESTDATA_DIR}/case_01_example_5.txt
    ${SOFTMAX_TESTDATA_DIR}/case_02_single_1.txt
    ${SOFTMAX_TESTDATA_DIR}/case_03_equal_32.txt
    ${SOFTMAX_TESTDATA_DIR}/case_04_warp_31.txt
    ${SOFTMAX_TESTDATA_DIR}/case_05_block_boundary_256.txt
    ${SOFTMAX_TESTDATA_DIR}/case_06_block_misaligned_257.txt
    ${SOFTMAX_TESTDATA_DIR}/case_07_large_positive_64.txt
    ${SOFTMAX_TESTDATA_DIR}/case_08_large_negative_64.txt
    ${SOFTMAX_TESTDATA_DIR}/case_09_stability_1024.txt
    ${SOFTMAX_TESTDATA_DIR}/case_10_perf_500000.txt
)

add_custom_command(
    OUTPUT ${SOFTMAX_TESTDATA_STAMP}
    BYPRODUCTS ${SOFTMAX_TESTDATA_FILES}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SOFTMAX_TESTDATA_DIR}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_test_data.py --out_dir ${SOFTMAX_TESTDATA_DIR}
    COMMAND ${CMAKE_COMMAND} -E touch ${SOFTMAX_TESTDATA_STAMP}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_test_data.py
    COMMENT "Generating softmax test data"
    VERBATIM
)

add_custom_target(softmax_testdata DEPENDS ${SOFTMAX_TESTDATA_STAMP})

add_executable(${SOFTMAX_TARGET} main.cu solve.cu)
set_target_properties(${SOFTMAX_TARGET} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
add_dependencies(${SOFTMAX_TARGET} softmax_testdata)

add_test(
    NAME softmax_compare
    COMMAND $<TARGET_FILE:${SOFTMAX_TARGET}> --test_dir ${SOFTMAX_TESTDATA_DIR}
)
set_tests_properties(softmax_compare PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_target(
    softmax_check
    COMMAND $<TARGET_FILE:${SOFTMAX_TARGET}> --test_dir ${SOFTMAX_TESTDATA_DIR}
    DEPENDS ${SOFTMAX_TARGET} softmax_testdata
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    USES_TERMINAL
    COMMENT "Running softmax compare"
)

if(LEETGPU_RUN_CHECK_ON_BUILD)
    add_custom_target(softmax_check_on_build ALL DEPENDS softmax_check)
endif()
