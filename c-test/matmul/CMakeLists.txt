################################################################################
# c-test: MatMul
#
# C wrapper tests of matmul kernel with SoftFloat Dialect.
################################################################################

project(test-matmul)

add_executable(
    ${PROJECT_NAME}
    test.cpp
    temp/test.o
)

target_link_libraries(
    ${PROJECT_NAME}
    softfloat
)
