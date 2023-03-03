add_mlir_conversion_library(Base2ToArith
    Base2ToArith.cpp

    DEPENDS
        Base2ConversionPassIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRDialectUtils
        MLIRArithDialect
        MLIRFuncDialect
        MLIRTransformUtils
        Base2Dialect
)
