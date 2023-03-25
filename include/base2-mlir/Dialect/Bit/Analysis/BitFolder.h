/// Declares the Bit dialect folding helper.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "base2-mlir/Dialect/Bit/Analysis/BitSequence.h"
#include "base2-mlir/Dialect/Bit/Enums.h"
#include "base2-mlir/Dialect/Bit/Interfaces/BitSequenceAttr.h"
#include "base2-mlir/Dialect/Bit/Interfaces/BitSequenceType.h"

namespace mlir::bit {

/// Singleton that implements Bit dialect operation folding.
class BitFolder {
public:
    /// Creates a constant boolean @p value .
    [[nodiscard]] static BitSequenceAttr makeBool(MLIRContext &ctx, bool value)
    {
        const auto i1Ty = IntegerType::get(&ctx, 1);
        return BitSequenceAttr::get(i1Ty, BitSequence(value));
    }
    /// Creates a constant false boolean value.
    [[nodiscard]] static BitSequenceAttr makeFalse(MLIRContext &ctx)
    {
        return makeBool(ctx, false);
    }
    /// Creates a constant true boolean value.
    [[nodiscard]] static BitSequenceAttr makeTrue(MLIRContext &ctx)
    {
        return makeBool(ctx, true);
    }

    /// Folds a bit cast.
    ///
    /// @pre    `in && resultTy`
    /// @pre    bit widths of @p in and @p resultTy match
    /// @pre    shapes of @p in and @p resultTy match
    [[nodiscard]] static BitSequenceLikeAttr
    bitCast(BitSequenceLikeAttr in, BitSequenceLikeType resultTy);

    /// Folds a bitwise comparison.
    ///
    /// @pre    `lhs && rhs`
    /// @pre    bit widths of @p lhs and @p rhs match
    /// @pre    shapes of @p lhs and @p rhs match
    [[nodiscard]] static BitSequenceLikeAttr bitCmp(
        EqualityPredicate predicate,
        BitSequenceLikeAttr lhs,
        BitSequenceLikeAttr rhs);

    /// Folds a bit sequence ternary operator.
    ///
    /// @pre    `condition && trueValue && falseValue`
    /// @pre    @p condition has an `i1` element type
    /// @pre    `trueValue.getElementType() == falseValue.getElementType()`
    /// @pre    shapes of @p trueValue and @p falseValue match
    /// @pre    @p condition is scalar, or matches the shape of @p trueValue
    [[nodiscard]] static BitSequenceLikeAttr bitSelect(
        BitSequenceLikeAttr condition,
        BitSequenceLikeAttr trueValue,
        BitSequenceLikeAttr falseValue);
    /// Folds a bit sequence ternary operator.
    ///
    /// @pre    `condition && trueValue && falseValue`
    /// @pre    @p condition has an `i1` element type
    /// @pre    @p trueValue and @p falseValue have the same element type
    /// @pre    shapes of @p trueValue and @p falseValue match
    /// @pre    @p condition is scalar, or matches the shape of @p trueValue
    [[nodiscard]] static OpFoldResult bitSelect(
        BitSequenceLikeAttr condition,
        OpFoldResult trueValue,
        OpFoldResult falseValue);

    /// Folds a bitwise complement operator.
    ///
    /// @pre    `value`
    [[nodiscard]] static BitSequenceLikeAttr bitCmpl(BitSequenceLikeAttr value);

    /// Folds a bitwise logical and operator.
    ///
    /// @pre    `lhs && rhs`
    /// @pre    `lhs.getType() == rhs.getType()`
    [[nodiscard]] static BitSequenceLikeAttr
    bitAnd(BitSequenceLikeAttr lhs, BitSequenceLikeAttr rhs);
    /// Folds a bitwise logical and operator.
    ///
    /// @pre    `lhs && rhs`
    /// @pre    @p lhs and @p rhs have the same element type
    [[nodiscard]] static OpFoldResult
    bitAnd(OpFoldResult lhs, BitSequenceLikeAttr rhs);

    /// Folds a bitwise logical or operator.
    ///
    /// @pre    `lhs && rhs`
    /// @pre    `lhs.getType() == rhs.getType()`
    [[nodiscard]] static BitSequenceLikeAttr
    bitOr(BitSequenceLikeAttr lhs, BitSequenceLikeAttr rhs);
    /// Folds a bitwise logical or operator.
    ///
    /// @pre    `lhs && rhs`
    /// @pre    @p lhs and @p rhs have the same element type
    [[nodiscard]] static OpFoldResult
    bitOr(OpFoldResult lhs, BitSequenceLikeAttr rhs);

    /// Folds a bitwise logical exclusive or operator.
    ///
    /// @pre    `lhs && rhs`
    /// @pre    `lhs.getType() == rhs.getType()`
    [[nodiscard]] static BitSequenceLikeAttr
    bitXor(BitSequenceLikeAttr lhs, BitSequenceLikeAttr rhs);
    /// Folds a bitwise logical exclusive or operator.
    ///
    /// @pre    `lhs && rhs`
    /// @pre    @p lhs and @p rhs have the same element type
    [[nodiscard]] static OpFoldResult
    bitXor(OpFoldResult lhs, BitSequenceLikeAttr rhs);

    /// Folds a left shift operator.
    ///
    /// @pre    `value`
    /// @pre    `!funnel || (value.getType() == funnel.getType())`
    [[nodiscard]] static BitSequenceLikeAttr bitShl(
        BitSequenceLikeAttr value,
        bit_width_t amount,
        BitSequenceLikeAttr funnel = {});
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
    [[nodiscard]] static BitSequenceLikeAttr bitShr(
        BitSequenceLikeAttr value,
        bit_width_t amount,
        BitSequenceLikeAttr funnel = {});
    /// Folds a right shift operator.
    ///
    /// @pre    `value`
    /// @pre    @p funnel is empty or has the same type as @p value
    [[nodiscard]] static OpFoldResult
    bitShr(OpFoldResult value, bit_width_t amount, OpFoldResult funnel = {});
};

} // namespace mlir::bit
