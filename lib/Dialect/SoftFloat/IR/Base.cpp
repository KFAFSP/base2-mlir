/// Implements the SoftFloat dialect base.
///
/// @file
/// @author     Jihaong Bi (jiahong.bi@mailbox.tu-dresden.de)

#include "base2-mlir/Dialect/SoftFloat/IR/Base.h"

#include "base2-mlir/Dialect/SoftFloat/IR/SoftFloat.h"

#define DEBUG_TYPE "softfloat-base"

using namespace mlir;
using namespace mlir::softfloat;

//===- Generated implementation -------------------------------------------===//

#include "base2-mlir/Dialect/SoftFloat/IR/Base.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// SoftFloatDialect
//===----------------------------------------------------------------------===//

void SoftFloatDialect::initialize()
{
    registerOps();
    registerTypes();
}
