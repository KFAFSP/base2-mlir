/// Implements the Base2 dialect base.
///
/// @file
/// @author     Jihaong Bi (jiahong.bi@mailbox.tu-dresden.de)
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "base2-mlir/Dialect/Base2/IR/Base.h"

#include "base2-mlir/Dialect/Base2/IR/Base2.h"
#include "mlir/Transforms/InliningUtils.h"

#include <memory>

using namespace mlir;
using namespace mlir::base2;

//===- Generated implementation -------------------------------------------===//

#include "base2-mlir/Dialect/Base2/IR/Base.cpp.inc"

//===----------------------------------------------------------------------===//

namespace {

struct Base2InlinerInterface : public DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;

    bool isLegalToInline(Operation*, Region*, bool, IRMapping &) const final
    {
        return true;
    }
};

} // namespace

//===----------------------------------------------------------------------===//
// Base2Dialect
//===----------------------------------------------------------------------===//

Operation* Base2Dialect::materializeConstant(
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

void Base2Dialect::initialize()
{
    registerAttributes();
    registerOps();
    registerTypes();

    // Implement interfaces for built-in types.
    registerBitSequenceAttrModels(*getContext());
    registerBitSequenceTypeModels(*getContext());
    registerFixedPointSemanticsModels(*getContext());
    registerInterpretableTypeModels(*getContext());
}
