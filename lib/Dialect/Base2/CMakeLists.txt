add_subdirectory(Analysis)
add_subdirectory(Interfaces)
add_subdirectory(IR)

add_library(Base2Enums OBJECT
    Enums.cpp
)
add_dependencies(Base2Enums Base2EnumsIncGen)
