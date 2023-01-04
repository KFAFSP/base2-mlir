add_mlir_library(Base2Analysis
        BitInterpreter.cpp
        BitSequence.cpp
        DynamicValue.cpp
        FixedPointInterpreter.cpp
        IntInterpreter.cpp

    DEPENDS
        Base2IncGen

    LINK_LIBS PUBLIC
        MLIRIR
        Base2Enums
)
