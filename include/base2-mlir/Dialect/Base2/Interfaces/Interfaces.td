//===- Interfaces.td ---------------------------------------*- tablegen -*-===//
//
// This is the convenience include for all Base2 dialect interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef BASE2_INTERFACES
#define BASE2_INTERFACES

include "base2-mlir/Dialect/Base2/Interfaces/ArithmeticOp.td"
include "base2-mlir/Dialect/Base2/Interfaces/FixedPointSemantics.td"
include "base2-mlir/Dialect/Base2/Interfaces/IEEE754Semantics.td"
include "base2-mlir/Dialect/Base2/Interfaces/InterpretableType.td"
include "base2-mlir/Dialect/Base2/Interfaces/RoundingOp.td"

#endif
