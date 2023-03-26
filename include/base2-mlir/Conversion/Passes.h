/// Declares the conversion passes.
///
/// @file
/// @author     Jihaong Bi (jiahong.bi@mailbox.tu-dresden.de)
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "base2-mlir/Conversion/BitToLLVM/BitToLLVM.h"

namespace mlir::base2 {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "base2-mlir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir::base2
