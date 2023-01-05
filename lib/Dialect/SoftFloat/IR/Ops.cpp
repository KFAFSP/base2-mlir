/// Implements the SoftFloat dialect ops.
///
/// @file
/// @author     Jihaong Bi (jiahong.bi@mailbox.tu-dresden.de)

#include "base2-mlir/Dialect/SoftFloat/IR/Ops.h"

#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/APFloat.h"

#define DEBUG_TYPE "softfloat-ops"

using namespace mlir;
using namespace mlir::softfloat;

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "base2-mlir/Dialect/SoftFloat/IR/Ops.cpp.inc"

//===----------------------------------------------------------------------===//
// SoftFloatDialect
//===----------------------------------------------------------------------===//

void SoftFloatDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "base2-mlir/Dialect/SoftFloat/IR/Ops.cpp.inc"
        >();
}
