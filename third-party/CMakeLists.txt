# The MLIR undefined behavior dialect.
add_subdirectory(ub-mlir EXCLUDE_FROM_ALL)
list(APPEND THIRD_PARTY_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}/ub-mlir/include
    ${CMAKE_CURRENT_BINARY_DIR}/ub-mlir/include)

list(JOIN THIRD_PARTY_INCLUDES ";" THIRD_PARTY_INCLUDES)
set(THIRD_PARTY_INCLUDES ${THIRD_PARTY_INCLUDES} PARENT_SCOPE)
