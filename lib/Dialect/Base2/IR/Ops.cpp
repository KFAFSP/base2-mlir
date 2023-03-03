/// Implements the Base2 dialect ops.
///
/// @file
/// @author     Jihaong Bi (jiahong.bi@mailbox.tu-dresden.de)
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "base2-mlir/Dialect/Base2/IR/Ops.h"

#include "base2-mlir/Dialect/Base2/Analysis/DynamicValue.h"
#include "base2-mlir/Dialect/Base2/Analysis/FixedPointInterpreter.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

#include <optional>

#define DEBUG_TYPE "base2-ops"

using namespace mlir;
using namespace mlir::base2;

/// Obtains a type with the same shape as @p type , using @p elementTy.
///
/// @pre    `type`
/// @pre    `elementTy`
[[nodiscard]] static Type getSameShape(Type type, Type elementTy)
{
    if (const auto shapedTy = type.dyn_cast<ShapedType>())
        return shapedTy.cloneWith(std::nullopt, elementTy);

    return elementTy;
}

/// Obtains the I1 or container-of-I1 that matches the shape of @p type .
///
/// @pre    `type`
[[nodiscard]] static Type getI1SameShape(Type type)
{
    return getSameShape(type, IntegerType::get(type.getContext(), 1));
}

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "base2-mlir/Dialect/Base2/IR/Ops.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

LogicalResult ConstantOp::inferReturnTypes(
    MLIRContext*,
    Optional<Location>,
    ValueRange,
    DictionaryAttr attributes,
    RegionRange,
    SmallVectorImpl<Type> &result)
{
    const auto value =
        attributes.getAs<BitSequenceLikeAttr>(getAttributeNames()[0]);
    if (!value) return failure();

    result.push_back(value.getType());
    return success();
}

namespace {

struct BuiltinConstant : OpRewritePattern<ConstantOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(ConstantOp op, PatternRewriter &rewriter) const override
    {
        // Construct the canonical attribute.
        const auto canonical = BitSequenceLikeAttr::get(op.getValue());

        // Attempt to update the value.
        if (op.getValue() == canonical) return failure();
        rewriter.updateRootInPlace(op, [&]() { op.setValueAttr(canonical); });
        return success();
    }
};

} // namespace

void ConstantOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns,
    MLIRContext* context)
{
    patterns.add<BuiltinConstant>(context);
}

//===----------------------------------------------------------------------===//
// BitCastOp
//===----------------------------------------------------------------------===//

bool BitCastOp::areCastCompatible(TypeRange inputs, TypeRange outputs)
{
    assert(inputs.size() == 1 && outputs.size() == 1);

    const auto inTy =
        getElementTypeOrSelf(inputs[0]).dyn_cast<BitSequenceType>();
    const auto outTy =
        getElementTypeOrSelf(outputs[0]).dyn_cast<BitSequenceType>();
    if (!inTy || !outTy) return false;

    return inTy.getBitWidth() == outTy.getBitWidth();
}

namespace {

// bitcast(bitcast(x : A) : B) -> bitcast(x : B)
struct TransitiveBitCastPattern : OpRewritePattern<BitCastOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(BitCastOp op, PatternRewriter &rewriter) const override
    {
        // Find the topmost BitCastOp in this tree.
        auto root = op;
        while (const auto parent = root.getIn().getDefiningOp<BitCastOp>())
            root = parent;

        // No change possible.
        if (op == root) return failure();

        // Directly cast from root input to desired output.
        rewriter.updateRootInPlace(op, [&]() {
            op.setOperand(root.getOperand());
        });
        return success();
    }
};

} // namespace

void BitCastOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns,
    MLIRContext* context)
{
    patterns.add<TransitiveBitCastPattern>(context);
}

//===----------------------------------------------------------------------===//
// ValueCastOp
//===----------------------------------------------------------------------===//

bool ValueCastOp::areCastCompatible(TypeRange inputs, TypeRange outputs)
{
    assert(inputs.size() == 1 && outputs.size() == 1);
    assert(inputs.front() && outputs.front());

    // Delegate to the BitInterpreter, which also tries symmetry.
    return BitInterpreter::canValueCast(inputs.front(), outputs.front());
}

//===----------------------------------------------------------------------===//
// Fixed-point operations
//===----------------------------------------------------------------------===//

template<class Fn>
[[nodiscard]] static LogicalResult inferFixedPointType(
    ValueRange operands,
    SmallVectorImpl<Type> &resultTypes,
    Fn fn)
{
    assert(operands.size() == 2);

    // Get the underlying FixedPointLikeType.
    const auto lhsTy =
        getElementTypeOrSelf(operands.front()).dyn_cast<FixedPointLikeType>();
    const auto rhsTy =
        getElementTypeOrSelf(operands.back()).dyn_cast<FixedPointLikeType>();
    if (!lhsTy || !rhsTy) return failure();

    // Get the FixedPointSemantics.
    const auto lhsSema = lhsTy.getSemantics();
    const auto rhsSema = rhsTy.getSemantics();

    // Infer the result semantics and return the canonical type.
    const auto outSema = fn(lhsSema, rhsSema);
    resultTypes.push_back(getSameShape(
        lhsTy,
        FixedPointType::get(
            outSema.getIntegerType(),
            outSema.getFractionalBits())));
    return success();
}

LogicalResult FixedAddOp::inferReturnTypes(
    MLIRContext*,
    Optional<Location>,
    ValueRange operands,
    DictionaryAttr,
    RegionRange,
    SmallVectorImpl<Type> &resultTypes)
{
    return inferFixedPointType(
        operands,
        resultTypes,
        [](auto &&lhs, auto &&rhs) {
            return FixedPointInterpreter::add(lhs, rhs);
        });
}

//===----------------------------------------------------------------------===//
// Base2Dialect
//===----------------------------------------------------------------------===//

void Base2Dialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "base2-mlir/Dialect/Base2/IR/Ops.cpp.inc"
        >();
}
