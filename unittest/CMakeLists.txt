################################################################################
# base2-mlir-tests
#
# The base2-mlir unit test project.
################################################################################

project(base2-mlir-tests)

find_package(doctest REQUIRED)
include(doctest)

include(CTest)

add_executable(${PROJECT_NAME}
    main.cpp
    BitSequence.cpp
)
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        BitDialect
)
target_compile_options(${PROJECT_NAME}
    PRIVATE
        -fno-rtti
)

doctest_discover_tests(${PROJECT_NAME})
