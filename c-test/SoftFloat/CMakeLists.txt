################################################################################
# c-test: SoftFloat
#
# C wrapper tests of operations within SoftFloat Dialect.
################################################################################

project(test-softfloat)

add_executable(
    ${PROJECT_NAME}
    test.cpp
    temp/test.o
)

target_link_libraries(
    ${PROJECT_NAME}
    softfloat
)
