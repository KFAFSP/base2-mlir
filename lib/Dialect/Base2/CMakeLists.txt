add_subdirectory(Analysis)
add_subdirectory(Interfaces)
add_subdirectory(IR)

add_mlir_library(Base2Enums
        Enums.cpp

    DEPENDS
        Base2EnumsIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRParser
)
