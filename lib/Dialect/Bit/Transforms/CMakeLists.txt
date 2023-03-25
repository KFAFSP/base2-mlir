add_mlir_library(BitTransforms
        LowerFunnelShift.cpp

    ENABLE_AGGREGATION
    DEPENDS
        BitPassesIncGen
    LINK_LIBS PUBLIC
        MLIRPass
        MLIRTransforms
        MLIRTransformUtils
)
