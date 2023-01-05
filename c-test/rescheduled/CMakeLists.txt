################################################################################
# c-test: SoftFloat
#
# C wrapper tests of operations within SoftFloat Dialect.
################################################################################

project(test-rescheduled)

add_executable(
    ${PROJECT_NAME}
    test.cpp
    temp/test_sf.o
    temp/test_std.o
)

target_link_libraries(
    ${PROJECT_NAME}
    softfloat
)
