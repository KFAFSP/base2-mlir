################################################################################
# base2-lsp-server
#
# The base2-lsp-server MLIR language server.
################################################################################

project(base2-lsp-server)

add_executable(${PROJECT_NAME}
    base2-lsp-server.cpp
)

# Link all standard MLIR dialect and conversion libs.
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        MLIRLspServerLib
        ${dialect_libs}
        ${conversion_libs}
)
