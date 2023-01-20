add_mlir_conversion_library(Base2ToSoftFloat
    Base2ToSoftFloat.cpp

    DEPENDS
        Base2ConversionPassIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRDialectUtils
        MLIRFuncDialect
        MLIRLinalgDialect
        MLIRLinalgTransforms
        MLIRTransformUtils
        Base2Dialect
        SoftFloatDialect
)
