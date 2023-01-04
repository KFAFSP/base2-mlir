################################################################################
# base2-opt
#
# The base2-mlir optimizer driver.
################################################################################

project(base2-opt)

add_executable(${PROJECT_NAME}
    base2-opt.cpp
)

# Link all standard MLIR dialect and conversion libs.
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        MLIROptLib
        ${dialect_libs}
        ${conversion_libs}
)
