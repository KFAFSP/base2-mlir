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

#endif // BASE2_CONVERSION_PASSES
