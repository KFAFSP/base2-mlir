/// Implements the Bit dialect base.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "base2-mlir/Dialect/Bit/IR/Base.h"

#include "base2-mlir/Dialect/Bit/IR/Bit.h"
#include "mlir/Transforms/InliningUtils.h"

#include <memory>

using namespace mlir;
using namespace mlir::bit;

//===- Generated implementation -------------------------------------------===//

#include "base2-mlir/Dialect/Bit/IR/Base.cpp.inc"

//===----------------------------------------------------------------------===//

namespace {

struct BitInlinerInterface : public DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;

    bool isLegalToInline(Operation*, Region*, bool, IRMapping &) const final
    {
        return true;
    }
};

} // namespace

//===----------------------------------------------------------------------===//
// BitDialect
//===----------------------------------------------------------------------===//

Operation* BitDialect::materializeConstant(
    OpBuilder &builder,
    Attribute value,
    Type type,
    Location location)
{
    // Materialize bit sequences using our constant op.
    if (const auto impl = value.dyn_cast<BitSequenceLikeAttr>()) {
        if (type != impl.getType()) return nullptr;
        return builder.create<ConstantOp>(location, impl);
    }

    return nullptr;
}

void BitDialect::initialize()
{
    registerAttributes();
    registerOps();

    // Implement interfaces for built-in types.
    registerBitSequenceAttrModels(*getContext());
    registerBitSequenceTypeModels(*getContext());
}
