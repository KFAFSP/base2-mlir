/// Declaration of the Base2 dialect ops.
///
/// @file
/// @author     Jihaong Bi (jiahong.bi@mailbox.tu-dresden.de)

#pragma once

#include "base2-mlir/Dialect/SoftFloat/IR/Types.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===- Generated includes -------------------------------------------------===//

#define GET_OP_CLASSES
#include "base2-mlir/Dialect/SoftFloat/IR/Ops.h.inc"

//===----------------------------------------------------------------------===//
