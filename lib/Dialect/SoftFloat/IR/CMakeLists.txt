add_mlir_dialect_library(SoftFloatDialect
        Base.cpp
        Ops.cpp
        Types.cpp

    DEPENDS
        SoftFloatIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces
)
