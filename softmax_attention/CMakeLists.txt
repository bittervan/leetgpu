find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(SOFTMAX_ATTENTION_TARGET softmax_attention_runner)
set(SOFTMAX_ATTENTION_TESTDATA_DIR ${CMAKE_CURRENT_BINARY_DIR}/testdata)
set(SOFTMAX_ATTENTION_TESTDATA_STAMP ${SOFTMAX_ATTENTION_TESTDATA_DIR}/.generated.stamp)
set(SOFTMAX_ATTENTION_TESTDATA_FILES
    ${SOFTMAX_ATTENTION_TESTDATA_DIR}/case_00_example_2x3x4.txt
    ${SOFTMAX_ATTENTION_TESTDATA_DIR}/case_01_single_1x1x1.txt
    ${SOFTMAX_ATTENTION_TESTDATA_DIR}/case_02_warp_m_31_n_32_d_8.txt
    ${SOFTMAX_ATTENTION_TESTDATA_DIR}/case_03_warp_n_32_m_32_d_16.txt
    ${SOFTMAX_ATTENTION_TESTDATA_DIR}/case_04_block_boundary_32x32x32.txt
    ${SOFTMAX_ATTENTION_TESTDATA_DIR}/case_05_block_misaligned_33x35x17.txt
    ${SOFTMAX_ATTENTION_TESTDATA_DIR}/case_06_perf_512x256x128.txt
)

add_custom_command(
    OUTPUT ${SOFTMAX_ATTENTION_TESTDATA_STAMP}
    BYPRODUCTS ${SOFTMAX_ATTENTION_TESTDATA_FILES}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SOFTMAX_ATTENTION_TESTDATA_DIR}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_test_data.py --out_dir ${SOFTMAX_ATTENTION_TESTDATA_DIR}
    COMMAND ${CMAKE_COMMAND} -E touch ${SOFTMAX_ATTENTION_TESTDATA_STAMP}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_test_data.py
    COMMENT "Generating softmax_attention test data"
    VERBATIM
)

add_custom_target(softmax_attention_testdata DEPENDS ${SOFTMAX_ATTENTION_TESTDATA_STAMP})

add_executable(${SOFTMAX_ATTENTION_TARGET} main.cu solve.cu)
set_target_properties(${SOFTMAX_ATTENTION_TARGET} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
add_dependencies(${SOFTMAX_ATTENTION_TARGET} softmax_attention_testdata)

add_test(
    NAME softmax_attention_compare
    COMMAND $<TARGET_FILE:${SOFTMAX_ATTENTION_TARGET}> --test_dir ${SOFTMAX_ATTENTION_TESTDATA_DIR}
)
set_tests_properties(softmax_attention_compare PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_target(
    softmax_attention_check
    COMMAND $<TARGET_FILE:${SOFTMAX_ATTENTION_TARGET}> --test_dir ${SOFTMAX_ATTENTION_TESTDATA_DIR}
    DEPENDS ${SOFTMAX_ATTENTION_TARGET} softmax_attention_testdata
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    USES_TERMINAL
    COMMENT "Running softmax_attention compare"
)

if(LEETGPU_RUN_CHECK_ON_BUILD)
    add_custom_target(softmax_attention_check_on_build ALL DEPENDS softmax_attention_check)
endif()
