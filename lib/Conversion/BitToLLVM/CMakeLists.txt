add_mlir_conversion_library(BitToLLVM
        BitToLLVM.cpp

    DEPENDS
        ConversionIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRDialectUtils
        MLIRTransformUtils
        MLIRLLVMDialect
        MLIRIndexDialect
        BitDialect
)
