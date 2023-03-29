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
    [[nodiscard]] static ValueOrPoisonLikeAttr
    bitCast(ValueOrPoisonLikeAttr in, BitSequenceLikeType resultTy);

    /// Folds a bitwise comparison.
    [[nodiscard]] static ConstOrPoison
    bitCmp(bool isEq, ConstOrPoison lhs, ConstOrPoison rhs)
    {
        if (!lhs || !rhs) return poison;
        return (*lhs == *rhs) == isEq;
    }
    /// Folds a bitwise comparison.
    ///
    /// @pre    `lhs && rhs`
    /// @pre    bit widths of @p lhs and @p rhs match
    /// @pre    shapes of @p lhs and @p rhs match
    [[nodiscard]] static ValueOrPoisonLikeAttr bitCmp(
        EqualityPredicate predicate,
        ValueOrPoisonLikeAttr lhs,
        ValueOrPoisonLikeAttr rhs);
    /// Folds a bitwise comparison.
    ///
    /// @pre    `lhs && rhs`
    /// @pre    bit widths of @p lhs and @p rhs match
    /// @pre    shapes of @p lhs and @p rhs match
    [[nodiscard]] static OpFoldResult
    bitCmp(EqualityPredicate predicate, OpFoldResult lhs, OpFoldResult rhs);

    /// Folds a bit sequence ternary operator.
    [[nodiscard]] static ConstOrPoison bitSelect(
        ConstOrPoison condition,
        ConstOrPoison trueValue,
        ConstOrPoison falseValue)
    {
        if (!condition) return poison;
        return condition->isZeros() ? falseValue : trueValue;
    }
    /// Folds a bit sequence ternary operator.
    ///
    /// @pre    `condition && trueValue && falseValue`
    /// @pre    @p condition has an `i1` element type
    /// @pre    `trueValue.getElementType() == falseValue.getElementType()`
    /// @pre    shapes of @p trueValue and @p falseValue match
    /// @pre    @p condition is scalar, or matches the shape of @p trueValue
    [[nodiscard]] static ValueOrPoisonLikeAttr bitSelect(
        ValueOrPoisonLikeAttr condition,
        ValueOrPoisonLikeAttr trueValue,
        ValueOrPoisonLikeAttr falseValue);
    /// Folds a bit sequence ternary operator.
    ///
    /// @pre    `condition && trueValue && falseValue`
    /// @pre    @p condition has an `i1` element type
    /// @pre    @p trueValue and @p falseValue have the same element type
    /// @pre    shapes of @p trueValue and @p falseValue match
    /// @pre    @p condition is scalar, or matches the shape of @p trueValue
    [[nodiscard]] static OpFoldResult bitSelect(
        ValueOrPoisonLikeAttr condition,
        OpFoldResult trueValue,
        OpFoldResult falseValue);
    /// Folds a bit sequence ternary operator.
    ///
    /// @pre    `condition && trueValue && falseValue`
    /// @pre    @p condition has an `i1` element type
    /// @pre    @p trueValue and @p falseValue have the same element type
    /// @pre    shapes of @p trueValue and @p falseValue match
    /// @pre    @p condition is scalar, or matches the shape of @p trueValue
    [[nodiscard]] static OpFoldResult bitSelect(
        OpFoldResult condition,
        OpFoldResult trueValue,
        OpFoldResult falseValue);

    /// Folds a bitwise logical and operator.
    [[nodiscard]] static ConstOrPoison
    bitAnd(ConstOrPoison lhs, ConstOrPoison rhs)
    {
        if (lhs && lhs->isZeros()) return *lhs;
        if (rhs && rhs->isZeros()) return *rhs;
        if (!lhs || !rhs) return poison;
        return lhs->logicAnd(*rhs);
    }
    /// Folds a bitwise logical and operator.
    ///
    /// @pre    `lhs && rhs`
    /// @pre    `lhs.getType() == rhs.getType()`
    [[nodiscard]] static ValueOrPoisonLikeAttr
    bitAnd(ValueOrPoisonLikeAttr lhs, ValueOrPoisonLikeAttr rhs);
    /// Folds a bitwise logical and operator.
    ///
    /// @pre    `lhs && rhs`
    /// @pre    @p lhs and @p rhs have the same type
    [[nodiscard]] static OpFoldResult
    bitAnd(OpFoldResult lhs, ValueOrPoisonLikeAttr rhs);

    /// Folds a bitwise logical or operator.
    [[nodiscard]] static ConstOrPoison
    bitOr(ConstOrPoison lhs, ConstOrPoison rhs)
    {
        if (lhs && lhs->isOnes()) return *lhs;
        if (rhs && rhs->isOnes()) return *rhs;
        if (!lhs || !rhs) return poison;
        return lhs->logicOr(*rhs);
    }
    /// Folds a bitwise logical or operator.
    ///
    /// @pre    `lhs && rhs`
    /// @pre    `lhs.getType() == rhs.getType()`
    [[nodiscard]] static ValueOrPoisonLikeAttr
    bitOr(ValueOrPoisonLikeAttr lhs, ValueOrPoisonLikeAttr rhs);
    /// Folds a bitwise logical or operator.
    ///
    /// @pre    `lhs && rhs`
    /// @pre    @p lhs and @p rhs have the same type
    [[nodiscard]] static OpFoldResult
    bitOr(OpFoldResult lhs, ValueOrPoisonLikeAttr rhs);

    /// Folds a bitwise logical exclusive or operator.
    [[nodiscard]] static ConstOrPoison
    bitXor(ConstOrPoison lhs, ConstOrPoison rhs)
    {
        if (!lhs || !rhs) return poison;
        return lhs->logicXor(*rhs);
    }
    /// Folds a bitwise logical exclusive or operator.
    ///
    /// @pre    `lhs && rhs`
    /// @pre    `lhs.getType() == rhs.getType()`
    [[nodiscard]] static ValueOrPoisonLikeAttr
    bitXor(ValueOrPoisonLikeAttr lhs, ValueOrPoisonLikeAttr rhs);
    /// Folds a bitwise logical exclusive or operator.
    ///
    /// @pre    `lhs && rhs`
    /// @pre    @p lhs and @p rhs have the same type
    [[nodiscard]] static OpFoldResult
    bitXor(OpFoldResult lhs, ValueOrPoisonLikeAttr rhs);

    /// Folds a left shift operator.
    [[nodiscard]] static ConstOrPoison
    bitShl(bit_width_t bitWidth, ConstOrPoison value, bit_width_t amount)
    {
        if (amount >= bitWidth) return Const::zeros(bitWidth);
        if (!value) return poison;
        return value->logicShl(amount);
    }
    /// Folds a left shift operator.
    [[nodiscard]] static ConstOrPoison bitShl(
        bit_width_t bitWidth,
        ConstOrPoison value,
        ConstOrPoison funnel,
        bit_width_t amount)
    {
        if (amount == 0) return value;
        if (amount >= 2 * bitWidth) return Const::zeros(bitWidth);
        if (amount < bitWidth && !value) return poison;
        if (!funnel) return poison;
        return value.value_or(*funnel).funnelShl(*funnel, amount);
    }
    /// Folds a left shift operator.
    ///
    /// @pre    `value`
    /// @pre    `!funnel || (value.getType() == funnel.getType())`
    [[nodiscard]] static ValueOrPoisonLikeAttr bitShl(
        ValueOrPoisonLikeAttr value,
        bit_width_t amount,
        ValueOrPoisonLikeAttr funnel = {});
    /// Folds a left shift operator.
    ///
    /// @pre    `value`
    /// @pre    @p funnel is empty or has the same type as @p value
    [[nodiscard]] static OpFoldResult
    bitShl(OpFoldResult value, bit_width_t amount, OpFoldResult funnel = {});
    /// Folds a left shift operator.
    ///
    /// @pre    `value && amount`
    /// @pre    `amount.getType().isIndex()`
    /// @pre    @p funnel is empty or has the same type as @p value
    [[nodiscard]] static OpFoldResult bitShl(
        OpFoldResult value,
        ub::ValueOrPoisonAttr<IntegerAttr> amount,
        OpFoldResult funnel = {});

    /// Folds a right shift operator.
    [[nodiscard]] static ConstOrPoison
    bitShr(bit_width_t bitWidth, ConstOrPoison value, bit_width_t amount)
    {
        if (amount >= bitWidth) return Const::zeros(bitWidth);
        if (!value) return poison;
        return value->logicShr(amount);
    }
    /// Folds a right shift operator.
    [[nodiscard]] static ConstOrPoison bitShr(
        bit_width_t bitWidth,
        ConstOrPoison value,
        ConstOrPoison funnel,
        bit_width_t amount)
    {
        if (amount == 0) return value;
        if (amount >= 2 * bitWidth) return Const::zeros(bitWidth);
        if (amount < bitWidth && !value) return poison;
        if (!funnel) return poison;
        return value.value_or(*funnel).funnelShr(*funnel, amount);
    }
    /// Folds a right shift operator.
    ///
    /// @pre    `value`
    /// @pre    `!funnel || (value.getType() == funnel.getType())`
    [[nodiscard]] static ValueOrPoisonLikeAttr bitShr(
        ValueOrPoisonLikeAttr value,
        bit_width_t amount,
        ValueOrPoisonLikeAttr funnel = {});
    /// Folds a right shift operator.
    ///
    /// @pre    `value`
    /// @pre    @p funnel is empty or has the same type as @p value
    [[nodiscard]] static OpFoldResult
    bitShr(OpFoldResult value, bit_width_t amount, OpFoldResult funnel = {});
    /// Folds a right shift operator.
    ///
    /// @pre    `value && amount`
    /// @pre    `amount.getType().isIndex()`
    /// @pre    @p funnel is empty or has the same type as @p value
    [[nodiscard]] static OpFoldResult bitShr(
        OpFoldResult value,
        ub::ValueOrPoisonAttr<IntegerAttr> amount,
        OpFoldResult funnel = {});

    /// Folds a population count operator.
    ///
    /// @pre    `value`
    [[nodiscard]] static ub::ValueOrPoisonAttr<IntegerAttr>
    bitCount(ValueOrPoisonAttr value)
    {
        if (value.isPoison())
            return ub::ValueOrPoisonAttr<IntegerAttr>::get(
                IndexType::get(value.getContext()));

        return IntegerAttr::get(
            IndexType::get(value.getContext()),
            value.getValueAttr().getValue().countOnes());
    }
    /// Folds a leading zero count operator.
    ///
    /// @pre    `value`
    [[nodiscard]] static ub::ValueOrPoisonAttr<IntegerAttr>
    bitClz(ValueOrPoisonAttr value)
    {
        if (value.isPoison())
            return ub::ValueOrPoisonAttr<IntegerAttr>::get(
                IndexType::get(value.getContext()));

        return IntegerAttr::get(
            IndexType::get(value.getContext()),
            value.getValueAttr().getValue().countLeadingZeros());
    }
    /// Folds a trailing zero count operator.
    ///
    /// @pre    `value`
    [[nodiscard]] static ub::ValueOrPoisonAttr<IntegerAttr>
    bitCtz(ValueOrPoisonAttr value)
    {
        if (value.isPoison())
            return ub::ValueOrPoisonAttr<IntegerAttr>::get(
                IndexType::get(value.getContext()));

        return IntegerAttr::get(
            IndexType::get(value.getContext()),
            value.getValueAttr().getValue().countTrailingZeros());
    }
};

} // namespace mlir::bit
