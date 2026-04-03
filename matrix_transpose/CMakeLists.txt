find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(MATRIX_TRANSPOSE_TARGET matrix_transpose_runner)
set(MATRIX_TRANSPOSE_TESTDATA_DIR ${CMAKE_CURRENT_BINARY_DIR}/testdata)
set(MATRIX_TRANSPOSE_TESTDATA_STAMP ${MATRIX_TRANSPOSE_TESTDATA_DIR}/.generated.stamp)
set(MATRIX_TRANSPOSE_TESTDATA_FILES
    ${MATRIX_TRANSPOSE_TESTDATA_DIR}/case_00_sanity_2x3.txt
    ${MATRIX_TRANSPOSE_TESTDATA_DIR}/case_01_single_1x1.txt
    ${MATRIX_TRANSPOSE_TESTDATA_DIR}/case_02_row_1x7.txt
    ${MATRIX_TRANSPOSE_TESTDATA_DIR}/case_03_col_7x1.txt
    ${MATRIX_TRANSPOSE_TESTDATA_DIR}/case_04_square_4x4.txt
    ${MATRIX_TRANSPOSE_TESTDATA_DIR}/case_05_rect_5x9.txt
    ${MATRIX_TRANSPOSE_TESTDATA_DIR}/case_06_tile_boundary_16x16.txt
    ${MATRIX_TRANSPOSE_TESTDATA_DIR}/case_07_tile_misaligned_17x19.txt
    ${MATRIX_TRANSPOSE_TESTDATA_DIR}/case_08_large_33x65.txt
)

add_custom_command(
    OUTPUT ${MATRIX_TRANSPOSE_TESTDATA_STAMP}
    BYPRODUCTS ${MATRIX_TRANSPOSE_TESTDATA_FILES}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${MATRIX_TRANSPOSE_TESTDATA_DIR}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_test_data.py --out_dir ${MATRIX_TRANSPOSE_TESTDATA_DIR}
    COMMAND ${CMAKE_COMMAND} -E touch ${MATRIX_TRANSPOSE_TESTDATA_STAMP}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_test_data.py
    COMMENT "Generating matrix_transpose test data"
    VERBATIM
)

add_custom_target(matrix_transpose_testdata DEPENDS ${MATRIX_TRANSPOSE_TESTDATA_STAMP})

add_executable(${MATRIX_TRANSPOSE_TARGET} main.cu solve.cu)
set_target_properties(${MATRIX_TRANSPOSE_TARGET} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
add_dependencies(${MATRIX_TRANSPOSE_TARGET} matrix_transpose_testdata)

add_test(
    NAME matrix_transpose_compare
    COMMAND $<TARGET_FILE:${MATRIX_TRANSPOSE_TARGET}> --test_dir ${MATRIX_TRANSPOSE_TESTDATA_DIR}
)
set_tests_properties(matrix_transpose_compare PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_target(
    matrix_transpose_check
    COMMAND $<TARGET_FILE:${MATRIX_TRANSPOSE_TARGET}> --test_dir ${MATRIX_TRANSPOSE_TESTDATA_DIR}
    DEPENDS ${MATRIX_TRANSPOSE_TARGET} matrix_transpose_testdata
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    USES_TERMINAL
    COMMENT "Running matrix_transpose compare"
)

if(LEETGPU_RUN_CHECK_ON_BUILD)
    add_custom_target(matrix_transpose_check_on_build ALL DEPENDS matrix_transpose_check)
endif()
