//===- Passes.td - Conversion passes ------------------------*- tablegen -*-===//
//
// This is the definitions file for the base2-mlir dialect conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef BASE2_MLIR_CONVERSION_PASSES
#define BASE2_MLIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertBitToLLVM : Pass<"convert-bit-to-llvm"> {
    let summary = "Convert Bit dialect to LLVM";

    let constructor = "mlir::createConvertBitToLLVMPass()";

    let dependentDialects = [
        "bit::BitDialect",
        "index::IndexDialect",
        "LLVM::LLVMDialect"
    ];

    let options = [
        Option<"indexBitwidth", "index-bitwidth", "unsigned",
            /*default=kDeriveIndexBitwidthFromDataLayout*/"0",
            "Bitwidth of the index type, 0 to use size of machine word">
    ];
}

#endif
