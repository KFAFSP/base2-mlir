add_mlir_dialect_library(BitDialect
        Attributes.cpp
        Base.cpp
        Folders.cpp
        Matchers.cpp
        Ops.cpp
        Types.cpp

    DEPENDS
        BitIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces
        MLIRIndexDialect
        UBDialect
        BitAnalysis
        BitEnums
        BitInterfaces
        BitTransforms
)
