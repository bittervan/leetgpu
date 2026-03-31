find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(VECTOR_ADD_TARGET vector_add_runner)
set(VECTOR_ADD_TESTDATA_DIR ${CMAKE_CURRENT_BINARY_DIR}/testdata)
set(VECTOR_ADD_TESTDATA_STAMP ${VECTOR_ADD_TESTDATA_DIR}/.generated.stamp)
set(VECTOR_ADD_TESTDATA_FILES
    ${VECTOR_ADD_TESTDATA_DIR}/case_00_sanity_4.txt
    ${VECTOR_ADD_TESTDATA_DIR}/case_01_single_1.txt
    ${VECTOR_ADD_TESTDATA_DIR}/case_02_small_7.txt
    ${VECTOR_ADD_TESTDATA_DIR}/case_03_warp_31.txt
    ${VECTOR_ADD_TESTDATA_DIR}/case_04_warp_boundary_32.txt
    ${VECTOR_ADD_TESTDATA_DIR}/case_05_block_edge_255.txt
    ${VECTOR_ADD_TESTDATA_DIR}/case_06_block_boundary_256.txt
    ${VECTOR_ADD_TESTDATA_DIR}/case_07_block_misaligned_257.txt
    ${VECTOR_ADD_TESTDATA_DIR}/case_08_large_1024.txt
)

add_custom_command(
    OUTPUT ${VECTOR_ADD_TESTDATA_STAMP}
    BYPRODUCTS ${VECTOR_ADD_TESTDATA_FILES}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${VECTOR_ADD_TESTDATA_DIR}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_test_data.py --out_dir ${VECTOR_ADD_TESTDATA_DIR}
    COMMAND ${CMAKE_COMMAND} -E touch ${VECTOR_ADD_TESTDATA_STAMP}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_test_data.py
    COMMENT "Generating vector_add test data"
    VERBATIM
)

add_custom_target(vector_add_testdata DEPENDS ${VECTOR_ADD_TESTDATA_STAMP})

add_executable(${VECTOR_ADD_TARGET} main.cu solve.cu)
set_target_properties(${VECTOR_ADD_TARGET} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
add_dependencies(${VECTOR_ADD_TARGET} vector_add_testdata)

add_test(
    NAME vector_add_compare
    COMMAND $<TARGET_FILE:${VECTOR_ADD_TARGET}> --test_dir ${VECTOR_ADD_TESTDATA_DIR}
)
set_tests_properties(vector_add_compare PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_target(
    vector_add_check
    COMMAND $<TARGET_FILE:${VECTOR_ADD_TARGET}> --test_dir ${VECTOR_ADD_TESTDATA_DIR}
    DEPENDS ${VECTOR_ADD_TARGET} vector_add_testdata
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    USES_TERMINAL
    COMMENT "Running vector_add compare"
)

if(LEETGPU_RUN_CHECK_ON_BUILD)
    add_custom_target(vector_add_check_on_build ALL DEPENDS vector_add_check)
endif()
