/// Implements IR matchers for the Bit dialect.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "base2-mlir/Dialect/Bit/IR/Matchers.h"

#include <numeric>

using namespace mlir;
using namespace mlir::bit;
using namespace mlir::bit::match;

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
[[nodiscard]] static BitSequence makeDefault(BitSequenceType elementTy)
{
    return BitSequence::zeros(elementTy.getBitWidth());
}

/// Make an std::optional<BitSequence> value from @p isPoison and @p value .
[[nodiscard]] static std::optional<BitSequence>
orPoison(bool isPoison, const BitSequence &value)
{
    if (isPoison) return poison;
    return value;
}

//===----------------------------------------------------------------------===//
// ConstOrPoison
//===----------------------------------------------------------------------===//

ConstOrPoison ConstOrPoison::map(
    function_ref<ConstOrPoison::UnaryFn> fn,
    BitSequenceType elementTy) const
{
    // Infer the type of the result.
    const auto inTy = getType();
    if (!elementTy) elementTy = inTy.getElementType();
    const auto outTy = inTy.getSameShape(elementTy);

    // Deal with fully poisoned values.
    if (isPoison()) return getSplat(outTy, fn(poison));

    // Prepare the poison masks.
    const auto numElements = getNumElements(inTy);
    assert(numElements != ShapedType::kDynamic);
    const auto inMask = getPoisonMask().zextOrTrunc(numElements);
    llvm::APInt outMask(numElements, 0UL);
    std::size_t idx = 0;

    // Compute the value attribute.
    const auto valueAttr = getValueAttr().map(
        [&](const auto &el) -> BitSequence {
            const auto result = fn(orPoison(inMask[idx], el));
            if (!result) outMask.setBit(idx);
            ++idx;
            return result.value_or(makeDefault(elementTy));
        },
        elementTy,
        false);

    return get(valueAttr, outMask);
}

[[nodiscard]] static ConstOrPoison zipImpl(
    function_ref<ConstOrPoison::BinaryFn> fn,
    ConstOrPoison lhs,
    ConstOrPoison rhs,
    BitSequenceType elementTy)
{
    assert(lhs && rhs);

    // Infer the type of the result.
    const auto inTy = lhs.getType().cast<BitSequenceLikeType>();
    if (!elementTy) elementTy = inTy.getElementType();
    const auto outTy = inTy.getSameShape(elementTy);

    // Deal with fully poisoned values.
    if (lhs.isPoison() && rhs.isPoison())
        return ConstOrPoison::getSplat(outTy, fn(poison, poison));

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
                const auto result = fn(orPoison(lhsMask[idx], el), poison);
                if (!result) outMask.setBit(idx);
                ++idx;
                return result.value_or(makeDefault(elementTy));
            },
            elementTy,
            false);

        return ConstOrPoison::get(valueAttr, outMask);
    }

    // Use binary zip.
    const auto rhsMask = rhs.getPoisonMask().zextOrTrunc(numElements);
    const auto valueAttr = lhs.getValueAttr().zip(
        [&](const auto &l, const auto &r) -> BitSequence {
            const auto result =
                fn(orPoison(lhsMask[idx], l), orPoison(rhsMask[idx], r));
            if (!result) outMask.setBit(idx);
            ++idx;
            return result.value_or(makeDefault(elementTy));
        },
        rhs.getValueAttr(),
        elementTy,
        false);

    return ConstOrPoison::get(valueAttr, outMask);
}

ConstOrPoison ConstOrPoison::zip(
    function_ref<ConstOrPoison::BinaryFn> fn,
    ConstOrPoison rhs,
    BitSequenceType elementTy) const
{
    return zipImpl(fn, *this, rhs, elementTy);
}

[[nodiscard]] static ConstOrPoison zipImpl(
    function_ref<ConstOrPoison::TernaryFn> fn,
    ConstOrPoison arg0,
    ConstOrPoison arg1,
    ConstOrPoison arg2,
    BitSequenceType elementTy)
{
    assert(arg0 && arg1 && arg2);

    // Infer the type of the result.
    const auto inTy = arg0.getType().cast<BitSequenceLikeType>();
    if (!elementTy) elementTy = inTy.getElementType();
    const auto outTy = inTy.getSameShape(elementTy);

    // Deal with fully poisoned values.
    if (arg0.isPoison() && arg1.isPoison() && arg2.isPoison())
        return ConstOrPoison::getSplat(outTy, fn(poison, poison, poison));

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
                const auto result =
                    fn(orPoison(mask0[idx], el), poison, poison);
                if (!result) outMask.setBit(idx);
                ++idx;
                return result.value_or(makeDefault(elementTy));
            },
            elementTy,
            false);

        return ConstOrPoison::get(valueAttr, outMask);
    }

    const auto mask1 = arg1.getPoisonMask().zextOrTrunc(numElements);
    if (arg2.isPoison()) {
        // Use binary zip instead of ternary zip.
        const auto valueAttr = arg0.getValueAttr().zip(
            [&](const auto &l, const auto &r) -> BitSequence {
                const auto result =
                    fn(orPoison(mask0[idx], l),
                       orPoison(mask1[idx], r),
                       poison);
                if (!result) outMask.setBit(idx);
                ++idx;
                return result.value_or(makeDefault(elementTy));
            },
            arg1.getValueAttr(),
            elementTy,
            false);

        return ConstOrPoison::get(valueAttr, outMask);
    }

    // Use ternary zip.
    const auto mask2 = arg2.getPoisonMask().zextOrTrunc(numElements);
    const auto valueAttr = arg0.getValueAttr().zip(
        [&](const auto &a, const auto &b, const auto &c) -> BitSequence {
            const auto result =
                fn(orPoison(mask0[idx], a),
                   orPoison(mask1[idx], b),
                   orPoison(mask2[idx], c));
            if (!result) outMask.setBit(idx);
            ++idx;
            return result.value_or(makeDefault(elementTy));
        },
        arg1.getValueAttr(),
        arg2.getValueAttr(),
        elementTy,
        false);

    return ConstOrPoison::get(valueAttr, outMask);
}

ConstOrPoison ConstOrPoison::zip(
    function_ref<ConstOrPoison::TernaryFn> fn,
    ConstOrPoison arg1,
    ConstOrPoison arg2,
    BitSequenceType elementTy) const
{
    return zipImpl(fn, *this, arg1, arg2, elementTy);
}
