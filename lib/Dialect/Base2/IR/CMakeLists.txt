add_mlir_dialect_library(Base2Dialect
        Attributes.cpp
        Base.cpp
        Ops.cpp
        Types.cpp

    DEPENDS
        Base2IncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces
        Base2Analysis
        Base2Enums
        Base2Interfaces
        BitDialect
)
