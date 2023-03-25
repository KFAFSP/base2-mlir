/// Declares the Bit ShiftOp interface.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "base2-mlir/Dialect/Bit/Interfaces/BitSequenceType.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir::bit {

/// Indicates the direction of a bit shift.
enum class ShiftDirection {
    /// Shift towards the MSB bit.
    TowardsMSB = 0,
    Left = 0,

    /// Shift towards the LSB bit.
    TowardsLSB = 1,
    Right = 1
};

} // namespace mlir::base2

//===- Generated includes -------------------------------------------------===//

#include "base2-mlir/Dialect/Bit/Interfaces/ShiftOp.h.inc"

//===----------------------------------------------------------------------===//
