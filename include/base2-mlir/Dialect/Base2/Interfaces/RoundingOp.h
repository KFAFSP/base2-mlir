/// Declares the Base2 RoundingOp interface.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "base2-mlir/Dialect/Base2/Enums.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir::base2::rounding_op_interface_defaults {

/// Queries the default attribute.
[[nodiscard]] RoundingMode getRoundingMode(Operation* self);

/// Sets the default attribute to RoundingMode::None.
void markAsKnownExact(Operation* self);

} // namespace mlir::base2::rounding_op_interface_defaults

//===- Generated includes -------------------------------------------------===//

#include "base2-mlir/Dialect/Base2/Interfaces/RoundingOp.h.inc"

//===----------------------------------------------------------------------===//
