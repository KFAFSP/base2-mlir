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

/// Unary ConstOrPoison functor.
using UnaryFn = function_ref<ConstOrPoison(ConstOrPoison)>;

ValueOrPoisonLikeAttr
map(UnaryFn fn, ValueOrPoisonLikeAttr attr, BitSequenceType elementTy = {});

/// Binary ConstOrPoison functor.
using BinaryFn = function_ref<ConstOrPoison(ConstOrPoison, ConstOrPoison)>;

ValueOrPoisonLikeAttr
zip(BinaryFn fn,
    ValueOrPoisonLikeAttr lhs,
    ValueOrPoisonAttr rhs,
    BitSequenceType elementTy = {});

/// Ternary ConstOrPoison functor.
using TernaryFn =
    function_ref<ConstOrPoison(ConstOrPoison, ConstOrPoison, ConstOrPoison)>;

ValueOrPoisonLikeAttr
zip(TernaryFn fn,
    ValueOrPoisonLikeAttr arg0,
    ValueOrPoisonLikeAttr arg1,
    ValueOrPoisonLikeAttr arg2,
    BitSequenceType elementTy = {});

} // namespace mlir::bit
