add_mlir_library(Base2Interfaces
        ArithmeticOp.cpp
        BitSequenceAttr.cpp
        BitSequenceType.cpp
        FixedPointSemantics.cpp
        InterpretableType.cpp
        RoundingOp.cpp

    DEPENDS
        Base2IncGen

    LINK_LIBS PUBLIC
        MLIRIR
        Base2Analysis
        Base2Enums
)
