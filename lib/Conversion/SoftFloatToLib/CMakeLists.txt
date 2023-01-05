add_mlir_conversion_library(SoftFloatToLib
        SoftFloatToLib.cpp

    DEPENDS
        Base2ConversionPassIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRDialectUtils
        MLIRFuncDialect
        MLIRTransformUtils
        SoftFloatDialect
)
