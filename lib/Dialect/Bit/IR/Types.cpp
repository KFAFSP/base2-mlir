/// Implements the Bit dialect types.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "base2-mlir/Dialect/Bit/IR/Types.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace mlir::bit;

//===- Generated implementation -------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "base2-mlir/Dialect/Bit/IR/Types.cpp.inc"

//===----------------------------------------------------------------------===//
