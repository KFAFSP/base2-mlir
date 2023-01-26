//===- Passes.td - Base2 dialect passes ---------------------*- tablegen -*-===//
//
// This is the definitions file for the Base2 dialect conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef BASE2_CONVERSION_PASSES
#define BASE2_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertSoftFloatToLib : Pass<"convert-softfloat-to-lib"> {
  let summary = "Convert SoftFloat dialect to lib softfloat calls";

  let constructor = "mlir::createConvertSoftFloatToLibPass()";

  let dependentDialects = [
    "AffineDialect",
    "arith::ArithDialect",
    "func::FuncDialect",
    "linalg::LinalgDialect",
    "memref::MemRefDialect",
  ];
}

def ConvertBase2ToSoftFloat : Pass<"convert-base2-to-softfloat"> {
  let summary = "Convert Base2 IEEE754 type operations to SoftFloat";

  let constructor = "mlir::createConvertBase2ToSoftFloatPass()";

  let dependentDialects = [
    "arith::ArithDialect",
    "func::FuncDialect",
    "linalg::LinalgDialect",
    "scf::SCFDialect",
    "softfloat::SoftFloatDialect",
    "tensor::TensorDialect",
  ];
}

def ConvertBase2ToArith : Pass<"convert-base2-to-ariht"> {
  let summary = "Convert Base2 to Arith operation";

  let constructor = "mlir::createConvertBase2ToArithPass()";

  let dependentDialects = [
    "arith::ArithDialect",
    "func::FuncDialect",
  ];
}

#endif // BASE2_CONVERSION_PASSES
