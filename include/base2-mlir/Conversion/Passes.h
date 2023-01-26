/// Declaration of the conversion pass within base2 dialect.
///
/// @file
/// @author     Jihaong Bi (jiahong.bi@mailbox.tu-dresden.de)

#pragma once

#include "base2-mlir/Conversion/Base2ToArith/Base2ToArith.h"
#include "base2-mlir/Conversion/Base2ToSoftFloat/Base2ToSoftFloat.h"
#include "base2-mlir/Conversion/SoftFloatToLib/SoftFloatToLib.h"

namespace mlir {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "base2-mlir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir