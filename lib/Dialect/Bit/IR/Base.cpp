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
    // Materialize poisoned constant values.
    if (const auto poisonAttr = value.dyn_cast<ub::PoisonAttr>())
        return builder.create<ub::PoisonOp>(location, poisonAttr);

    // Materialize constants of IndexType.
    // NOTE: This is one of the workarounds for IntegerAttr also encoding
    //       IndexType values while being a BitSequenceAttr.
    if (type.isa<IndexType>())
        return builder.create<index::ConstantOp>(
            location,
            value.cast<IntegerAttr>());

    // Materialize bit sequences using our constant op.
    if (auto impl = value.dyn_cast<BitSequenceLikeAttr>()) {
        if (type != impl.getType()) return nullptr;
        // Canonicalize the attribute for good measure.
        impl = BitSequenceLikeAttr::get(impl);
        return builder.create<ConstantOp>(location, impl);
    }

    return nullptr;
}

void BitDialect::initialize()
{
    registerAttributes();
    registerOps();

    // Implement the inliner interface.
    addInterfaces<BitInlinerInterface>();

    // Implement interfaces for built-in types.
    registerBitSequenceAttrModels(*getContext());
    registerBitSequenceTypeModels(*getContext());
}
