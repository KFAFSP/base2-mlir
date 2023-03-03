//===- Passes.td - Base2 dialect passes ---------------------*- tablegen -*-===//
//
// This is the definitions file for the Base2 dialect conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef BASE2_CONVERSION_PASSES
#define BASE2_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertBase2ToArith : Pass<"convert-base2-to-arith"> {
  let summary = "Convert Base2 to Arith operation";

  let constructor = "mlir::createConvertBase2ToArithPass()";

  let dependentDialects = [
    "arith::ArithDialect",
    "func::FuncDialect",
  ];
}

#endif // BASE2_CONVERSION_PASSES
