/// Implements the Base2 dialect attributes.
///
/// @file
/// @author     Jihaong Bi (jiahong.bi@mailbox.tu-dresden.de)
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "base2-mlir/Dialect/Base2/IR/Attributes.h"

#include "base2-mlir/Dialect/Base2/IR/Base2.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::base2;

//===- Generated implementation -------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "base2-mlir/Dialect/Base2/IR/Attributes.cpp.inc"

//===----------------------------------------------------------------------===//
