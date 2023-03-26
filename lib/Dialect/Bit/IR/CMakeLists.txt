add_mlir_dialect_library(BitDialect
        Attributes.cpp
        Base.cpp
        Ops.cpp
        Types.cpp

    DEPENDS
        BitIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces
        MLIRIndexDialect
        BitAnalysis
        BitEnums
        BitInterfaces
        BitTransforms
)
