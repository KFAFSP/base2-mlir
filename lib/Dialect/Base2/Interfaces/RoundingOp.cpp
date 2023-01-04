/// Implements the Base2 dialect RoundingOp interface.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "base2-mlir/Dialect/Base2/Interfaces/RoundingOp.h"

#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::base2;

RoundingMode rounding_op_interface_defaults::getRoundingMode(Operation* self)
{
    assert(self);

    if (const auto attr = self->getAttrOfType<RoundingModeAttr>(
            RoundingOp::getRoundingModeAttrName()))
        return attr.getValue();

    return RoundingMode::None;
}

void rounding_op_interface_defaults::markAsKnownExact(Operation* self)
{
    assert(self);

    self->setAttr(
        RoundingOp::getRoundingModeAttrName(),
        RoundingModeAttr::get(self->getContext(), RoundingMode::None));
}

//===- Generated implementation -------------------------------------------===//

#include "base2-mlir/Dialect/Base2/Interfaces/RoundingOp.cpp.inc"

//===----------------------------------------------------------------------===//
