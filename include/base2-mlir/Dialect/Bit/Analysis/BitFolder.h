/// Declares the Bit dialect folding helper.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "base2-mlir/Dialect/Bit/Analysis/PoisonSema.h"
#include "base2-mlir/Dialect/Bit/Enums.h"

namespace mlir::bit {

/// Singleton that implements Bit dialect operation folding.
class BitFolder {
public:
    /// Creates a constant poison value.
    ///
    /// @pre    `type`
    [[nodiscard]] static ub::PoisonAttr makePoison(Type type)
    {
        return ub::PoisonAttr::get(type);
    }

    /// Creates a constant boolean @p value .
    [[nodiscard]] static ValueAttr makeBool(MLIRContext &ctx, bool value)
    {
        const auto i1Ty = IntegerType::get(&ctx, 1);
        return BitSequenceAttr::get(i1Ty, BitSequence(value));
    }
    /// Creates a constant false boolean value.
    [[nodiscard]] static ValueAttr makeFalse(MLIRContext &ctx)
    {
        return makeBool(ctx, false);
    }
    /// Creates a constant true boolean value.
    [[nodiscard]] static ValueAttr makeTrue(MLIRContext &ctx)
    {
        return makeBool(ctx, true);
    }

    /// Folds a bit cast.
    ///
    /// @pre    `in && resultTy`
    /// @pre    bit widths of @p in and @p resultTy match
    /// @pre    shapes of @p in and @p resultTy match
    [[nodiscard]] static ValueLikeAttr
    bitCast(ValueLikeAttr in, BitSequenceLikeType resultTy);

    /// Folds a bitwise comparison.
    ///
    /// @pre    `lhs && rhs`
    /// @pre    bit widths of @p lhs and @p rhs match
    /// @pre    shapes of @p lhs and @p rhs match
    [[nodiscard]] static ValueLikeAttr
    bitCmp(EqualityPredicate predicate, ValueLikeAttr lhs, ValueLikeAttr rhs);

    /// Folds a bit sequence ternary operator.
    ///
    /// @pre    `condition && trueValue && falseValue`
    /// @pre    @p condition has an `i1` element type
    /// @pre    `trueValue.getElementType() == falseValue.getElementType()`
    /// @pre    shapes of @p trueValue and @p falseValue match
    /// @pre    @p condition is scalar, or matches the shape of @p trueValue
    [[nodiscard]] static ValueLikeAttr bitSelect(
        ValueLikeAttr condition,
        ValueLikeAttr trueValue,
        ValueLikeAttr falseValue);
    /// Folds a bit sequence ternary operator.
    ///
    /// @pre    `condition && trueValue && falseValue`
    /// @pre    @p condition has an `i1` element type
    /// @pre    @p trueValue and @p falseValue have the same element type
    /// @pre    shapes of @p trueValue and @p falseValue match
    /// @pre    @p condition is scalar, or matches the shape of @p trueValue
    [[nodiscard]] static OpFoldResult bitSelect(
        ValueLikeAttr condition,
        OpFoldResult trueValue,
        OpFoldResult falseValue);

    /// Folds a bitwise complement operator.
    ///
    /// @pre    `value`
    [[nodiscard]] static ValueLikeAttr bitCmpl(ValueLikeAttr value);

    /// Folds a bitwise logical and operator.
    ///
    /// @pre    `lhs && rhs`
    /// @pre    `lhs.getType() == rhs.getType()`
    [[nodiscard]] static ValueLikeAttr
    bitAnd(ValueLikeAttr lhs, ValueLikeAttr rhs);
    /// Folds a bitwise logical and operator.
    ///
    /// @pre    `lhs && rhs`
    /// @pre    @p lhs and @p rhs have the same element type
    [[nodiscard]] static OpFoldResult
    bitAnd(OpFoldResult lhs, ValueLikeAttr rhs);

    /// Folds a bitwise logical or operator.
    ///
    /// @pre    `lhs && rhs`
    /// @pre    `lhs.getType() == rhs.getType()`
    [[nodiscard]] static ValueLikeAttr
    bitOr(ValueLikeAttr lhs, ValueLikeAttr rhs);
    /// Folds a bitwise logical or operator.
    ///
    /// @pre    `lhs && rhs`
    /// @pre    @p lhs and @p rhs have the same element type
    [[nodiscard]] static OpFoldResult
    bitOr(OpFoldResult lhs, ValueLikeAttr rhs);

    /// Folds a bitwise logical exclusive or operator.
    ///
    /// @pre    `lhs && rhs`
    /// @pre    `lhs.getType() == rhs.getType()`
    [[nodiscard]] static ValueLikeAttr
    bitXor(ValueLikeAttr lhs, ValueLikeAttr rhs);
    /// Folds a bitwise logical exclusive or operator.
    ///
    /// @pre    `lhs && rhs`
    /// @pre    @p lhs and @p rhs have the same element type
    [[nodiscard]] static OpFoldResult
    bitXor(OpFoldResult lhs, ValueLikeAttr rhs);

    /// Folds a left shift operator.
    ///
    /// @pre    `value`
    /// @pre    `!funnel || (value.getType() == funnel.getType())`
    [[nodiscard]] static ValueLikeAttr
    bitShl(ValueLikeAttr value, bit_width_t amount, ValueLikeAttr funnel = {});
    /// Folds a left shift operator.
    ///
    /// @pre    `value`
    /// @pre    @p funnel is empty or has the same type as @p value
    [[nodiscard]] static OpFoldResult
    bitShl(OpFoldResult value, bit_width_t amount, OpFoldResult funnel = {});

    /// Folds a right shift operator.
    ///
    /// @pre    `value`
    /// @pre    `!funnel || (value.getType() == funnel.getType())`
    [[nodiscard]] static ValueLikeAttr
    bitShr(ValueLikeAttr value, bit_width_t amount, ValueLikeAttr funnel = {});
    /// Folds a right shift operator.
    ///
    /// @pre    `value`
    /// @pre    @p funnel is empty or has the same type as @p value
    [[nodiscard]] static OpFoldResult
    bitShr(OpFoldResult value, bit_width_t amount, OpFoldResult funnel = {});

    /// Folds a population count operator.
    ///
    /// @pre    `value`
    [[nodiscard]] static IntegerAttr bitCount(ValueAttr value)
    {
        return IntegerAttr::get(
            IndexType::get(value.getContext()),
            value.getValue().countOnes());
    }
    /// Folds a leading zero count operator.
    ///
    /// @pre    `value`
    [[nodiscard]] static IntegerAttr bitClz(ValueAttr value)
    {
        return IntegerAttr::get(
            IndexType::get(value.getContext()),
            value.getValue().countLeadingZeros());
    }
    /// Folds a trailing zero count operator.
    ///
    /// @pre    `value`
    [[nodiscard]] static IntegerAttr bitCtz(ValueAttr value)
    {
        return IntegerAttr::get(
            IndexType::get(value.getContext()),
            value.getValue().countTrailingZeros());
    }
};

} // namespace mlir::bit
