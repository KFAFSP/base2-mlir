/// Implements poison semantics for the Bit dialect.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "base2-mlir/Dialect/Bit/Analysis/PoisonSema.h"

#include <numeric>

using namespace mlir;
using namespace mlir::bit;

/// Gets the number of elements contained in @p type .
///
/// For a scalar type, this is 1. For a container type, this is the product of
/// its dimensions. For dynamic shapes, this is ShapedType::kDynamic;
[[nodiscard]] static std::int64_t getNumElements(Type type)
{
    if (const auto shapedTy = type.dyn_cast<ShapedType>()) {
        if (!shapedTy.hasStaticShape()) return ShapedType::kDynamic;

        return std::accumulate(
            shapedTy.getShape().begin(),
            shapedTy.getShape().end(),
            std::int64_t(1),
            std::multiplies<std::int64_t>{});
    }

    return 1;
}

/// Make a valid default value for @p elementTy .
///
/// @pre    `elementTy`
[[nodiscard]] static Const makeDefault(BitSequenceType elementTy)
{
    return BitSequence::zeros(elementTy.getBitWidth());
}

/// Make a result attribute for @p splatValue and @p resultTy .
///
/// @pre    `resultTy`
[[nodiscard]] static ValueOrPoisonLikeAttr
makeResult(ConstOrPoison splatValue, BitSequenceLikeType resultTy)
{
    assert(resultTy);

    // Make a fully poisoned value.
    if (!splatValue) return ValueOrPoisonLikeAttr::get(resultTy);

    // Make an unpoisoned value.
    return BitSequenceLikeAttr::getSplat(resultTy, *splatValue);
}

/// Make a result attribute for @p value and @p poisonMask .
///
/// @pre    `value`
[[nodiscard]] static ValueOrPoisonLikeAttr
makeResult(ValueLikeAttr value, const llvm::APInt &poisonMask)
{
    assert(value);

    // Handle quick canonicalization cases.
    if (poisonMask.isAllOnes())
        return ValueOrPoisonLikeAttr::get(value.getType());
    if (poisonMask.isZero()) return value;

    // Make a partially poisoned value.
    return ValueOrPoisonLikeAttr::get("bit", value, poisonMask);
}

/// Make a ConstOrPoison value from @p isPoison and @p value .
[[nodiscard]] static ConstOrPoison select(bool isPoison, const Const &value)
{
    if (isPoison) return poison;
    return value;
}

//===----------------------------------------------------------------------===//
// Constant folding
//===----------------------------------------------------------------------===//

ValueOrPoisonLikeAttr mlir::bit::map(
    UnaryFn fn,
    ValueOrPoisonLikeAttr attr,
    BitSequenceType elementTy)
{
    assert(attr);

    // Infer the type of the result.
    const auto inTy = attr.getType().cast<BitSequenceLikeType>();
    if (!elementTy) elementTy = inTy.getElementType();
    const auto outTy = inTy.getSameShape(elementTy);

    // Deal with fully poisoned values.
    if (attr.isPoison()) return makeResult(fn(poison), outTy);

    // Prepare the poison masks.
    const auto numElements = getNumElements(inTy);
    assert(numElements != ShapedType::kDynamic);
    const auto inMask = attr.getPoisonMask().zextOrTrunc(numElements);
    llvm::APInt outMask(numElements, 0UL);
    std::size_t idx = 0;

    // Compute the value attribute.
    const auto valueAttr = attr.getValueAttr().map(
        [&](const auto &el) -> BitSequence {
            const auto result = fn(select(inMask[idx], el));
            if (!result) outMask.setBit(idx);
            ++idx;
            return result.value_or(makeDefault(elementTy));
        },
        elementTy);

    return makeResult(valueAttr, outMask);
}

ValueOrPoisonLikeAttr mlir::bit::zip(
    BinaryFn fn,
    ValueOrPoisonLikeAttr lhs,
    ValueOrPoisonLikeAttr rhs,
    BitSequenceType elementTy)
{
    assert(lhs && rhs);

    // Infer the type of the result.
    const auto inTy = lhs.getType().cast<BitSequenceLikeType>();
    if (!elementTy) elementTy = inTy.getElementType();
    const auto outTy = inTy.getSameShape(elementTy);

    // Deal with fully poisoned values.
    if (lhs.isPoison() && rhs.isPoison())
        return makeResult(fn(poison, poison), outTy);

    // Make sure lhs is not fully poisoned.
    if (lhs.isPoison()) {
        std::swap(lhs, rhs);
        fn = [&](const auto &l, const auto &r) { return fn(r, l); };
    }

    // Prepare the poison masks.
    const auto numElements = getNumElements(inTy);
    assert(numElements != ShapedType::kDynamic);
    const auto lhsMask = lhs.getPoisonMask().zextOrTrunc(numElements);
    llvm::APInt outMask(numElements, 0UL);
    std::size_t idx = 0;

    if (rhs.isPoison()) {
        // Use map instead of zip.
        const auto valueAttr = lhs.getValueAttr().map(
            [&](const auto &el) -> BitSequence {
                const auto result = fn(select(lhsMask[idx], el), poison);
                if (!result) outMask.setBit(idx);
                ++idx;
                return result.value_or(makeDefault(elementTy));
            },
            elementTy);

        return makeResult(valueAttr, outMask);
    }

    // Use binary zip.
    const auto rhsMask = rhs.getPoisonMask().zextOrTrunc(numElements);
    const auto valueAttr = lhs.getValueAttr().zip(
        [&](const auto &l, const auto &r) -> BitSequence {
            const auto result =
                fn(select(lhsMask[idx], l), select(rhsMask[idx], r));
            if (!result) outMask.setBit(idx);
            ++idx;
            return result.value_or(makeDefault(elementTy));
        },
        rhs.getValueAttr(),
        elementTy);

    return makeResult(valueAttr, outMask);
}

ValueOrPoisonLikeAttr mlir::bit::zip(
    TernaryFn fn,
    ValueOrPoisonLikeAttr arg0,
    ValueOrPoisonLikeAttr arg1,
    ValueOrPoisonLikeAttr arg2,
    BitSequenceType elementTy)
{
    assert(arg0 && arg1 && arg2);

    // Infer the type of the result.
    const auto inTy = arg0.getType().cast<BitSequenceLikeType>();
    if (!elementTy) elementTy = inTy.getElementType();
    const auto outTy = inTy.getSameShape(elementTy);

    // Deal with fully poisoned values.
    if (arg0.isPoison() && arg1.isPoison() && arg2.isPoison())
        return makeResult(fn(poison, poison, poison), outTy);

    // Make sure arg0 is not fully poisoned.
    if (arg0.isPoison()) {
        if (!arg1.isPoison()) {
            std::swap(arg0, arg1);
            fn = [&](const auto &a, const auto &b, const auto &c) {
                return fn(b, a, c);
            };
        } else {
            assert(!arg2.isPoison());
            std::swap(arg0, arg2);
            fn = [&](const auto &a, const auto &b, const auto &c) {
                return fn(c, b, a);
            };
        }
    }

    // Try to make arg1 not fully poisoned.
    if (arg1.isPoison() && !arg2.isPoison()) {
        std::swap(arg1, arg2);
        fn = [&](const auto &a, const auto &b, const auto &c) {
            return fn(a, c, b);
        };
    }

    // Prepare the poison masks.
    const auto numElements = getNumElements(inTy);
    assert(numElements != ShapedType::kDynamic);
    const auto mask0 = arg0.getPoisonMask().zextOrTrunc(numElements);
    llvm::APInt outMask(numElements, 0UL);
    std::size_t idx = 0;

    if (arg1.isPoison() && arg2.isPoison()) {
        // Use map instead of zip.
        const auto valueAttr = arg0.getValueAttr().map(
            [&](const auto &el) -> BitSequence {
                const auto result = fn(select(mask0[idx], el), poison, poison);
                if (!result) outMask.setBit(idx);
                ++idx;
                return result.value_or(makeDefault(elementTy));
            },
            elementTy);

        return makeResult(valueAttr, outMask);
    }

    const auto mask1 = arg1.getPoisonMask().zextOrTrunc(numElements);
    if (arg2.isPoison()) {
        // Use binary zip instead of ternary zip.
        const auto valueAttr = arg0.getValueAttr().zip(
            [&](const auto &l, const auto &r) -> BitSequence {
                const auto result =
                    fn(select(mask0[idx], l), select(mask1[idx], r), poison);
                if (!result) outMask.setBit(idx);
                ++idx;
                return result.value_or(makeDefault(elementTy));
            },
            arg1.getValueAttr(),
            elementTy);

        return makeResult(valueAttr, outMask);
    }

    // Use ternary zip.
    const auto mask2 = arg2.getPoisonMask().zextOrTrunc(numElements);
    const auto valueAttr = arg0.getValueAttr().zip(
        [&](const auto &a, const auto &b, const auto &c) -> BitSequence {
            const auto result =
                fn(select(mask0[idx], a),
                   select(mask1[idx], b),
                   select(mask2[idx], c));
            if (!result) outMask.setBit(idx);
            ++idx;
            return result.value_or(makeDefault(elementTy));
        },
        arg1.getValueAttr(),
        arg2.getValueAttr(),
        elementTy);

    return makeResult(valueAttr, outMask);
}
