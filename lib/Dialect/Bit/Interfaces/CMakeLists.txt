add_mlir_library(BitInterfaces
        BitSequenceAttr.cpp
        BitSequenceType.cpp
        LogicOp.cpp
        ShiftOp.cpp

    ENABLE_AGGREGATION
    DEPENDS
        BitIncGen
)
