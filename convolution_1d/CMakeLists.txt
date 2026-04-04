find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(CONVOLUTION_1D_TARGET convolution_1d_runner)
set(CONVOLUTION_1D_TESTDATA_DIR ${CMAKE_CURRENT_BINARY_DIR}/testdata)
set(CONVOLUTION_1D_TESTDATA_STAMP ${CONVOLUTION_1D_TESTDATA_DIR}/.generated.stamp)
set(CONVOLUTION_1D_TESTDATA_FILES
    ${CONVOLUTION_1D_TESTDATA_DIR}/case_00_sanity_5_3.txt
    ${CONVOLUTION_1D_TESTDATA_DIR}/case_01_single_1_1.txt
    ${CONVOLUTION_1D_TESTDATA_DIR}/case_02_full_overlap_7_7.txt
    ${CONVOLUTION_1D_TESTDATA_DIR}/case_03_output_31.txt
    ${CONVOLUTION_1D_TESTDATA_DIR}/case_04_output_32.txt
    ${CONVOLUTION_1D_TESTDATA_DIR}/case_05_output_255.txt
    ${CONVOLUTION_1D_TESTDATA_DIR}/case_06_output_256.txt
    ${CONVOLUTION_1D_TESTDATA_DIR}/case_07_output_257.txt
    ${CONVOLUTION_1D_TESTDATA_DIR}/case_08_large_4096_31.txt
)

add_custom_command(
    OUTPUT ${CONVOLUTION_1D_TESTDATA_STAMP}
    BYPRODUCTS ${CONVOLUTION_1D_TESTDATA_FILES}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CONVOLUTION_1D_TESTDATA_DIR}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_test_data.py --out_dir ${CONVOLUTION_1D_TESTDATA_DIR}
    COMMAND ${CMAKE_COMMAND} -E touch ${CONVOLUTION_1D_TESTDATA_STAMP}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_test_data.py
    COMMENT "Generating convolution_1d test data"
    VERBATIM
)

add_custom_target(convolution_1d_testdata DEPENDS ${CONVOLUTION_1D_TESTDATA_STAMP})

add_executable(${CONVOLUTION_1D_TARGET} main.cu solve.cu)
set_target_properties(${CONVOLUTION_1D_TARGET} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
add_dependencies(${CONVOLUTION_1D_TARGET} convolution_1d_testdata)

add_test(
    NAME convolution_1d_compare
    COMMAND $<TARGET_FILE:${CONVOLUTION_1D_TARGET}> --test_dir ${CONVOLUTION_1D_TESTDATA_DIR}
)
set_tests_properties(convolution_1d_compare PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_target(
    convolution_1d_check
    COMMAND $<TARGET_FILE:${CONVOLUTION_1D_TARGET}> --test_dir ${CONVOLUTION_1D_TESTDATA_DIR}
    DEPENDS ${CONVOLUTION_1D_TARGET} convolution_1d_testdata
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    USES_TERMINAL
    COMMENT "Running convolution_1d compare"
)

if(LEETGPU_RUN_CHECK_ON_BUILD)
    add_custom_target(convolution_1d_check_on_build ALL DEPENDS convolution_1d_check)
endif()
