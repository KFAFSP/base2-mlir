add_subdirectory(Analysis)
add_subdirectory(Interfaces)
add_subdirectory(IR)

add_library(BitEnums OBJECT
        Enums.cpp
)
add_dependencies(BitEnums BitEnumsIncGen)
