/// Implements the Base2 dialect ops.
///
/// @file
/// @author     Jihaong Bi (jiahong.bi@mailbox.tu-dresden.de)
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "base2-mlir/Dialect/Base2/IR/Ops.h"

#include "base2-mlir/Dialect/Base2/Analysis/DynamicValue.h"
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

OpFoldResult ConstantOp::fold(FoldAdaptor) { return getValue(); }

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

OpFoldResult BitCastOp::fold(FoldAdaptor adaptor)
{
    assert(adaptor.getOperands().size() == 1);

    // Fold no-op casts.
    if (getType() == getIn().getType()) return getIn();

    // Fold constant bit casts.
    if (const auto attr =
            adaptor.getIn().dyn_cast_or_null<BitSequenceLikeAttr>())
        return attr.bitCastElements(getType().getElementType());

    // Otherwise folding is not performed.
    return OpFoldResult{};
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

OpFoldResult ValueCastOp::fold(FoldAdaptor adaptor)
{
    assert(adaptor.getOperands().size() == 1);

    // Fold no-op casts.
    if (getType() == getIn().getType()) return getIn();

    // Attempt folding using the BitInterpreter.
    return BitInterpreter::valueCast(
        adaptor.getIn(),
        getType(),
        getRoundingMode());
}

//===----------------------------------------------------------------------===//
// Common operations
//===----------------------------------------------------------------------===//

/// Folds a binary operation using @p fn on DynamicValue instances.
///
/// @pre    `variables.size() == 2`
/// @pre    `constants.size() == 2`
static auto
foldBinOp(auto fn, OperandRange variables, ArrayRef<Attribute> constants)
{
    assert(variables.size() == 2);
    assert(constants.size() == 2);

    return fn(
        constants.front()
            ? DynamicValue(constants.front().cast<BitSequenceLikeAttr>())
            : DynamicValue(variables.front()),
        constants.back()
            ? DynamicValue(constants.back().cast<BitSequenceLikeAttr>())
            : DynamicValue(variables.back()));
}

OpFoldResult CmpOp::fold(FoldAdaptor adaptor)
{
    const auto makeSplat = [&](bool value) -> Attribute {
        if (const auto shapedTy = getType().dyn_cast<ShapedType>())
            return DenseIntElementsAttr::get(shapedTy, value);
        return BoolAttr::get(getContext(), value);
    };
    const auto uiZeroCmp =
        [](PartialOrderingPredicate pred) -> std::optional<bool> {
        switch (pred) {
        case PartialOrderingPredicate::OrderedAndGreaterOrEqual:
        case PartialOrderingPredicate::UnorderedOrGreaterOrEqual: return true;
        case PartialOrderingPredicate::OrderedAndLess:
        case PartialOrderingPredicate::UnorderedOrLess: return false;
        default: return std::nullopt;
        }
    };

    // Verum and Falsum must always fold.
    switch (getPredicate()) {
    case PartialOrderingPredicate::Falsum: return makeSplat(false);
    case PartialOrderingPredicate::Verum: return makeSplat(true);
    default: break;
    }

    return foldBinOp(
        [&](DynamicValue lhs, DynamicValue rhs) -> OpFoldResult {
            // Special folding for fixed-point types.
            if (const auto fixedTy =
                    lhs.getElementType().dyn_cast<FixedPointSemantics>()) {
                // Ordered and Unordered must always fold.
                switch (getPredicate()) {
                case PartialOrderingPredicate::Ordered: return makeSplat(true);
                case PartialOrderingPredicate::Unordered:
                    return makeSplat(false);
                default: break;
                }

                // Comparing unsigned with zero also has special cases.
                if (fixedTy.isUnsigned()) {
                    std::optional<bool> zeroCmp;
                    if (rhs.isZero())
                        zeroCmp = uiZeroCmp(getPredicate());
                    else if (lhs.isZero())
                        zeroCmp = uiZeroCmp(flip(getPredicate()));
                    if (zeroCmp) return makeSplat(*zeroCmp);
                }
            }

            return lhs.cmp(getPredicate(), rhs);
        },
        getOperands(),
        adaptor.getOperands());
}

OpFoldResult MinOp::fold(FoldAdaptor adaptor)
{
    assert(adaptor.getOperands().size() == 2);

    return foldBinOp(
        [&](DynamicValue lhs, DynamicValue rhs) -> OpFoldResult {
            // Special folding for fixed-point types.
            if (const auto fixedTy =
                    lhs.getElementType().dyn_cast<FixedPointSemantics>()) {
                if (fixedTy.isUnsigned()) {
                    if (lhs.isZero()) return lhs;
                    if (rhs.isZero()) return rhs;
                }
            }

            return lhs.min(rhs);
        },
        getOperands(),
        adaptor.getOperands());
}

OpFoldResult MaxOp::fold(FoldAdaptor adaptor)
{
    assert(adaptor.getOperands().size() == 2);

    return foldBinOp(
        [&](DynamicValue lhs, DynamicValue rhs) -> OpFoldResult {
            // Special folding for fixed-point types.
            if (const auto fixedTy =
                    lhs.getElementType().dyn_cast<FixedPointSemantics>()) {
                if (fixedTy.isUnsigned()) {
                    if (lhs.isZero()) return rhs;
                    if (rhs.isZero()) return lhs;
                }
            }

            return lhs.max(rhs);
        },
        getOperands(),
        adaptor.getOperands());
}

//===----------------------------------------------------------------------===//
// Closed arithmetic operations
//===----------------------------------------------------------------------===//

OpFoldResult AddOp::fold(FoldAdaptor adaptor)
{
    assert(adaptor.getOperands().size() == 2);

    return foldBinOp(
        [&](auto &&lhs, auto &&rhs) { return lhs.add(rhs, getRoundingMode()); },
        getOperands(),
        adaptor.getOperands());
}

OpFoldResult SubOp::fold(FoldAdaptor adaptor)
{
    assert(adaptor.getOperands().size() == 2);

    return foldBinOp(
        [&](auto &&lhs, auto &&rhs) { return lhs.sub(rhs, getRoundingMode()); },
        getOperands(),
        adaptor.getOperands());
}

OpFoldResult MulOp::fold(FoldAdaptor adaptor)
{
    assert(adaptor.getOperands().size() == 2);

    return foldBinOp(
        [&](auto &&lhs, auto &&rhs) { return lhs.mul(rhs, getRoundingMode()); },
        getOperands(),
        adaptor.getOperands());
}

OpFoldResult DivOp::fold(FoldAdaptor adaptor)
{
    assert(adaptor.getOperands().size() == 2);

    return foldBinOp(
        [&](auto &&lhs, auto &&rhs) { return lhs.div(rhs, getRoundingMode()); },
        getOperands(),
        adaptor.getOperands());
}

OpFoldResult ModOp::fold(FoldAdaptor adaptor)
{
    assert(adaptor.getOperands().size() == 2);

    return foldBinOp(
        [&](auto &&lhs, auto &&rhs) { return lhs.mod(rhs); },
        getOperands(),
        adaptor.getOperands());
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
