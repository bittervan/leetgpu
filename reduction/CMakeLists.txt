find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(REDUCTION_TARGET reduction_runner)
set(REDUCTION_TESTDATA_DIR ${CMAKE_CURRENT_BINARY_DIR}/testdata)
set(REDUCTION_TESTDATA_STAMP ${REDUCTION_TESTDATA_DIR}/.generated.stamp)
set(REDUCTION_TESTDATA_FILES
    ${REDUCTION_TESTDATA_DIR}/case_00_example_8.txt
    ${REDUCTION_TESTDATA_DIR}/case_01_cancel_4.txt
    ${REDUCTION_TESTDATA_DIR}/case_02_single_1.txt
    ${REDUCTION_TESTDATA_DIR}/case_03_warp_31.txt
    ${REDUCTION_TESTDATA_DIR}/case_04_warp_boundary_32.txt
    ${REDUCTION_TESTDATA_DIR}/case_05_block_edge_255.txt
    ${REDUCTION_TESTDATA_DIR}/case_06_block_boundary_256.txt
    ${REDUCTION_TESTDATA_DIR}/case_07_block_misaligned_257.txt
    ${REDUCTION_TESTDATA_DIR}/case_08_large_4096.txt
    ${REDUCTION_TESTDATA_DIR}/case_09_perf_4194304.txt
)

add_custom_command(
    OUTPUT ${REDUCTION_TESTDATA_STAMP}
    BYPRODUCTS ${REDUCTION_TESTDATA_FILES}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${REDUCTION_TESTDATA_DIR}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_test_data.py --out_dir ${REDUCTION_TESTDATA_DIR}
    COMMAND ${CMAKE_COMMAND} -E touch ${REDUCTION_TESTDATA_STAMP}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_test_data.py
    COMMENT "Generating reduction test data"
    VERBATIM
)

add_custom_target(reduction_testdata DEPENDS ${REDUCTION_TESTDATA_STAMP})

add_executable(${REDUCTION_TARGET} main.cu solve.cu)
set_target_properties(${REDUCTION_TARGET} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
add_dependencies(${REDUCTION_TARGET} reduction_testdata)

add_test(
    NAME reduction_compare
    COMMAND $<TARGET_FILE:${REDUCTION_TARGET}> --test_dir ${REDUCTION_TESTDATA_DIR}
)
set_tests_properties(reduction_compare PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_target(
    reduction_check
    COMMAND $<TARGET_FILE:${REDUCTION_TARGET}> --test_dir ${REDUCTION_TESTDATA_DIR}
    DEPENDS ${REDUCTION_TARGET} reduction_testdata
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    USES_TERMINAL
    COMMENT "Running reduction compare"
)

if(LEETGPU_RUN_CHECK_ON_BUILD)
    add_custom_target(reduction_check_on_build ALL DEPENDS reduction_check)
endif()
