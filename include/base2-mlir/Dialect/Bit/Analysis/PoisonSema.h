/// Declares poison semantics for the Bit dialect.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "base2-mlir/Dialect/Bit/Analysis/BitSequence.h"
#include "base2-mlir/Dialect/Bit/Interfaces/BitSequenceAttr.h"
#include "base2-mlir/Dialect/Bit/Interfaces/BitSequenceType.h"
#include "ub-mlir/Dialect/UB/IR/UB.h"

namespace mlir::bit {

//===----------------------------------------------------------------------===//
// Type aliases
//===----------------------------------------------------------------------===//

/// Type that stores a constant value.
using Const = BitSequence;
/// Type that stores poison or a Value.
using ConstOrPoison = std::optional<Const>;

/// ConstOrPoison value that indicate poison.
static constexpr auto poison = std::nullopt;

/// Attribute that stores a compile-time constant bit sequence.
using ValueAttr = BitSequenceAttr;
/// Attribute that stores a single or container of ValueAttr.
using ValueLikeAttr = BitSequenceLikeAttr;
/// Attribute that stores poison or a ValueAttr.
using ValueOrPoisonAttr = ub::ValueOrPoisonAttr<ValueAttr>;
/// Attribute that stores poison or a ValueLikeAttr.
using ValueOrPoisonLikeAttr = ub::ValueOrPoisonAttr<ValueLikeAttr>;

//===----------------------------------------------------------------------===//
// Constant folding
//===----------------------------------------------------------------------===//

/// Reference to a unary ConstOrPoison functor.
using UnaryFn = function_ref<ConstOrPoison(ConstOrPoison)>;

/// Applies @p fn to @p attr .
///
/// If @p elementTy is nullptr, `attr.getType().getElementType()` is used.
///
/// @pre    `attr`
/// @pre    bit width of @p elementTy and @p fn result matches
[[nodiscard]] ValueOrPoisonLikeAttr
map(UnaryFn fn, ValueOrPoisonLikeAttr attr, BitSequenceType elementTy = {});

/// Reference to a binary ConstOrPoison functor.
using BinaryFn = function_ref<ConstOrPoison(ConstOrPoison, ConstOrPoison)>;

/// Combines the values contained in @p lhs and @p rhs using @p fn .
///
/// If @p elementTy is nullptr, `lhs.getType().getElementType()` is used.
///
/// @pre    `lhs && rhs`
/// @pre    shapes of @p lhs and @p rhs match
/// @pre    bit width of @p elementTy and @p fn result matches
[[nodiscard]] ValueOrPoisonLikeAttr
zip(BinaryFn fn,
    ValueOrPoisonLikeAttr lhs,
    ValueOrPoisonLikeAttr rhs,
    BitSequenceType elementTy = {});

/// Reference to a ternary ConstOrPoison functor.
using TernaryFn =
    function_ref<ConstOrPoison(ConstOrPoison, ConstOrPoison, ConstOrPoison)>;

/// Combines the values contained in @p arg0 , @p arg1 and @p arg2 using @p fn .
///
/// If @p elementTy is nullptr, `arg0.getType().getElementType()` is used.
///
/// @pre    `arg0 && arg1 && arg2`
/// @pre    shapes of @p arg0 , @p arg1 and @p arg2 match
/// @pre    bit width of @p elementTy and @p fn result matches
[[nodiscard]] ValueOrPoisonLikeAttr
zip(TernaryFn fn,
    ValueOrPoisonLikeAttr arg0,
    ValueOrPoisonLikeAttr arg1,
    ValueOrPoisonLikeAttr arg2,
    BitSequenceType elementTy = {});

} // namespace mlir::bit
