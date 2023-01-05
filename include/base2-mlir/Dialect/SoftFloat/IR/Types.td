//===- Types.td - SoftFloat dialect types ------------------*- tablegen -*-===//
//
// This is the definitions file for the SoftFloat dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef SOFTFLOAT_TYPES
#define SOFTFLOAT_TYPES

include "base2-mlir/Dialect/SoftFloat/IR/Attributes.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

def SoftFloat_SFloatType
        : SoftFloat_Type<"SFloat", [MemRefElementTypeInterface]> {
    let mnemonic = "sfloat";
    let summary = "strong name for __float64 type in softfloat lib";

    let genStorageClass = 0;
    let genVerifyDecl = 1;
}

#endif
