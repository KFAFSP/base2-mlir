/// Implements the Base2 dialect ops.
///
/// @file
/// @author     Jihaong Bi (jiahong.bi@mailbox.tu-dresden.de)
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "base2-mlir/Dialect/Base2/IR/Ops.h"

#include "base2-mlir/Dialect/Base2/Analysis/DynamicValue.h"
#include "base2-mlir/Utils.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

#include <optional>

#define DEBUG_TYPE "base2-ops"

using namespace mlir;
using namespace mlir::base2;
using namespace mlir::ext;

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "base2-mlir/Dialect/Base2/IR/Ops.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs)
{
    assert(inputs.size() == 1 && outputs.size() == 1);
    assert(inputs.front() && outputs.front());

    // Delegate to the BitInterpreter, which also tries symmetry.
    return BitInterpreter::canValueCast(inputs.front(), outputs.front());
}

OpFoldResult CastOp::fold(CastOp::FoldAdaptor adaptor)
{
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
            ? DynamicValue(constants.front().cast<bit::BitSequenceLikeAttr>())
            : DynamicValue(variables.front()),
        constants.back()
            ? DynamicValue(constants.back().cast<bit::BitSequenceLikeAttr>())
            : DynamicValue(variables.back()));
}

OpFoldResult CmpOp::fold(CmpOp::FoldAdaptor adaptor)
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

OpFoldResult MinOp::fold(MinOp::FoldAdaptor adaptor)
{
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

OpFoldResult MaxOp::fold(MaxOp::FoldAdaptor adaptor)
{
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

OpFoldResult AddOp::fold(AddOp::FoldAdaptor adaptor)
{
    return foldBinOp(
        [&](auto &&lhs, auto &&rhs) { return lhs.add(rhs, getRoundingMode()); },
        getOperands(),
        adaptor.getOperands());
}

OpFoldResult SubOp::fold(SubOp::FoldAdaptor adaptor)
{
    return foldBinOp(
        [&](auto &&lhs, auto &&rhs) { return lhs.sub(rhs, getRoundingMode()); },
        getOperands(),
        adaptor.getOperands());
}

OpFoldResult MulOp::fold(MulOp::FoldAdaptor adaptor)
{
    return foldBinOp(
        [&](auto &&lhs, auto &&rhs) { return lhs.mul(rhs, getRoundingMode()); },
        getOperands(),
        adaptor.getOperands());
}

OpFoldResult DivOp::fold(DivOp::FoldAdaptor adaptor)
{
    return foldBinOp(
        [&](auto &&lhs, auto &&rhs) { return lhs.div(rhs, getRoundingMode()); },
        getOperands(),
        adaptor.getOperands());
}

OpFoldResult ModOp::fold(ModOp::FoldAdaptor adaptor)
{
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
