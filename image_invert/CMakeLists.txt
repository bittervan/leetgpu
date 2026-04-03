find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(IMAGE_INVERT_TARGET image_invert_runner)
set(IMAGE_INVERT_TESTDATA_DIR ${CMAKE_CURRENT_BINARY_DIR}/testdata)
set(IMAGE_INVERT_TESTDATA_STAMP ${IMAGE_INVERT_TESTDATA_DIR}/.generated.stamp)
set(IMAGE_INVERT_TESTDATA_FILES
    ${IMAGE_INVERT_TESTDATA_DIR}/case_00_example_1x2.txt
    ${IMAGE_INVERT_TESTDATA_DIR}/case_01_example_2x1.txt
    ${IMAGE_INVERT_TESTDATA_DIR}/case_02_single_1x1.txt
    ${IMAGE_INVERT_TESTDATA_DIR}/case_03_row_7x1.txt
    ${IMAGE_INVERT_TESTDATA_DIR}/case_04_col_1x7.txt
    ${IMAGE_INVERT_TESTDATA_DIR}/case_05_rect_17x15_255_pixels.txt
    ${IMAGE_INVERT_TESTDATA_DIR}/case_06_square_16x16_256_pixels.txt
    ${IMAGE_INVERT_TESTDATA_DIR}/case_07_row_257x1_257_pixels.txt
    ${IMAGE_INVERT_TESTDATA_DIR}/case_08_rect_19x17_323_pixels.txt
    ${IMAGE_INVERT_TESTDATA_DIR}/case_09_large_65x33.txt
)

add_custom_command(
    OUTPUT ${IMAGE_INVERT_TESTDATA_STAMP}
    BYPRODUCTS ${IMAGE_INVERT_TESTDATA_FILES}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${IMAGE_INVERT_TESTDATA_DIR}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_test_data.py --out_dir ${IMAGE_INVERT_TESTDATA_DIR}
    COMMAND ${CMAKE_COMMAND} -E touch ${IMAGE_INVERT_TESTDATA_STAMP}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_test_data.py
    COMMENT "Generating image_invert test data"
    VERBATIM
)

add_custom_target(image_invert_testdata DEPENDS ${IMAGE_INVERT_TESTDATA_STAMP})

add_executable(${IMAGE_INVERT_TARGET} main.cu solve.cu)
set_target_properties(${IMAGE_INVERT_TARGET} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
add_dependencies(${IMAGE_INVERT_TARGET} image_invert_testdata)

add_test(
    NAME image_invert_compare
    COMMAND $<TARGET_FILE:${IMAGE_INVERT_TARGET}> --test_dir ${IMAGE_INVERT_TESTDATA_DIR}
)
set_tests_properties(image_invert_compare PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_target(
    image_invert_check
    COMMAND $<TARGET_FILE:${IMAGE_INVERT_TARGET}> --test_dir ${IMAGE_INVERT_TESTDATA_DIR}
    DEPENDS ${IMAGE_INVERT_TARGET} image_invert_testdata
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    USES_TERMINAL
    COMMENT "Running image_invert compare"
)

if(LEETGPU_RUN_CHECK_ON_BUILD)
    add_custom_target(image_invert_check_on_build ALL DEPENDS image_invert_check)
endif()
