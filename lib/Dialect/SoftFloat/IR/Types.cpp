/// Implements the SoftFloat dialect types.
///
/// @file
/// @author     Jihaong Bi (jiahong.bi@mailbox.tu-dresden.de)

#include "base2-mlir/Dialect/SoftFloat/IR/Types.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "softfloat-types"

using namespace mlir;
using namespace mlir::softfloat;

//===- Generated implementation -------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "base2-mlir/Dialect/SoftFloat/IR/Types.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// SoftFloatDialect
//===----------------------------------------------------------------------===//

void SoftFloatDialect::registerTypes()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "base2-mlir/Dialect/SoftFloat/IR/Types.cpp.inc"
        >();
}
