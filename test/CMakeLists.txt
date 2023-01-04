################################################################################
# check-base2-mlir
#
# The base2-mlir regression test project.
################################################################################

project(check-base2-mlir)

# Configure the testing site configuration.
configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)
set(TEST_DEPENDS
    FileCheck count not
    base2-opt
)

# Create the test suite.
add_lit_testsuite(${PROJECT_NAME}
    "Running the base2-mlir regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS
        ${TEST_DEPENDS}
)
add_lit_testsuites(BASE2MLIR
    ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS
        ${TEST_DEPENDS}
)
