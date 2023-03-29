/// Implements the Bit dialect folding helper.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "base2-mlir/Dialect/Bit/Analysis/BitFolder.h"

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace mlir::bit;

/// Gets the MLIRContext of @p value .
///
/// @pre    `value`
[[nodiscard]] static MLIRContext* getContext(OpFoldResult value)
{
    if (const auto attr = value.dyn_cast<Attribute>()) return attr.getContext();
    return value.dyn_cast<Value>().getContext();
}
/// Gets the type of @p value .
///
/// @pre    `value`
[[nodiscard]] static Type getType(OpFoldResult value)
{
    if (const auto attr = value.dyn_cast<Attribute>())
        return attr.cast<TypedAttr>().getType();
    return value.dyn_cast<Value>().getType();
}

/// Gets the splat value of @p attr , if any.
///
/// @pre   `attr`
[[nodiscard]] static std::optional<Const> getSplat(ValueLikeAttr attr)
{
    assert(attr);

    if (const auto single = attr.dyn_cast<ValueAttr>())
        return single.getValue();

    const auto dense = attr.cast<DenseBitSequencesAttr>();
    if (dense.isSplat()) return dense.getSplatValue();

    return std::nullopt;
}

//===----------------------------------------------------------------------===//
// BitFolder
//===----------------------------------------------------------------------===//

ValueOrPoisonLikeAttr
BitFolder::bitCast(ValueOrPoisonLikeAttr in, BitSequenceLikeType resultTy)
{
    assert(in && resultTy);
    assert(
        in.getType().getElementType().getBitWidth()
        == resultTy.getElementType().getBitWidth());
    assert(succeeded(verifyCompatibleShape(in.getType(), resultTy)));

    // NOTE: Instead of bitCastElements, we must now use map so that poison is
    //       handled correctly.
    return map(
        [](const auto &el) { return el; },
        in,
        resultTy.getElementType());
}

ValueOrPoisonLikeAttr BitFolder::bitCmp(
    EqualityPredicate predicate,
    ValueOrPoisonLikeAttr lhs,
    ValueOrPoisonLikeAttr rhs)
{
    assert(lhs && rhs);
    assert(
        lhs.getType().getElementType().getBitWidth()
        == rhs.getType().getElementType().getBitWidth());
    assert(succeeded(verifyCompatibleShape(lhs.getType(), rhs.getType())));

    // Zip the operands together.
    return zip(
        [=](const auto &l, const auto &r) {
            return bitCmp(predicate == EqualityPredicate::Equal, l, r);
        },
        lhs,
        rhs,
        IntegerType::get(lhs.getContext(), 1));
}

OpFoldResult BitFolder::bitCmp(
    EqualityPredicate predicate,
    OpFoldResult lhs,
    OpFoldResult rhs)
{
    assert(lhs && rhs);

    // Infer the result type.
    const auto inTy = getType(lhs).cast<BitSequenceLikeType>();
    const auto i1Ty = IntegerType::get(getContext(lhs), 1);
    const auto outTy = inTy.getSameShape(i1Ty);

    // Handle trivial predicate.
    switch (predicate) {
    case EqualityPredicate::Verum: return ValueLikeAttr::getSplat(outTy, true);
    case EqualityPredicate::Falsum:
        return ValueLikeAttr::getSplat(outTy, false);
    default: break;
    }

    // Handle poisoned case.
    if (ub::isPoison(lhs) || ub::isPoison(rhs))
        return ValueOrPoisonLikeAttr::get(outTy);

    // Handle trivial case.
    if (lhs == rhs)
        return ValueLikeAttr::getSplat(outTy, matches(true, predicate));

    // Handle constant case.
    const auto lhsAttr =
        lhs.dyn_cast<Attribute>().dyn_cast_or_null<ValueOrPoisonLikeAttr>();
    const auto rhsAttr =
        rhs.dyn_cast<Attribute>().dyn_cast_or_null<ValueOrPoisonLikeAttr>();
    if (lhsAttr && rhsAttr) return bitCmp(predicate, lhsAttr, rhsAttr);

    return {};
}

ValueOrPoisonLikeAttr BitFolder::bitSelect(
    ValueOrPoisonLikeAttr condition,
    ValueOrPoisonLikeAttr trueValue,
    ValueOrPoisonLikeAttr falseValue)
{
    assert(condition && trueValue && falseValue);
    assert(condition.getType().getElementType().isSignlessInteger(1));
    assert(trueValue.getType() == falseValue.getType());

    if (!condition.isPoisoned()) {
        if (auto splat = getSplat(condition.getValueAttr())) {
            // Quick exit if the condition is just a bool.
            return splat->isZeros() ? falseValue : trueValue;
        }
    }

    assert(succeeded(
        verifyCompatibleShape(condition.getType(), trueValue.getType())));

    // Perform a three-way zip.
    return zip(
        [](const auto &c, const auto &t, const auto &f) {
            return bitSelect(c, t, f);
        },
        condition,
        trueValue,
        falseValue,
        trueValue.getType().getElementType());
}

OpFoldResult BitFolder::bitSelect(
    ValueOrPoisonLikeAttr condition,
    OpFoldResult trueValue,
    OpFoldResult falseValue)
{
    assert(condition && trueValue && falseValue);

    // Handle trivial equality.
    if (trueValue == falseValue) return trueValue;

    // Handle poisoned condition.
    if (condition.isPoison())
        return ValueOrPoisonLikeAttr::get(
            getType(trueValue).cast<BitSequenceLikeType>());

    if (!condition.isPoisoned()) {
        if (const auto splat = getSplat(condition.getValueAttr())) {
            // Quick exit if the condition is just a bool.
            return splat->isZeros() ? falseValue : trueValue;
        }
    }

    // Fold if all values are constant.
    const auto trueAttr = trueValue.dyn_cast<Attribute>()
                              .dyn_cast_or_null<ValueOrPoisonLikeAttr>();
    const auto falseAttr = falseValue.dyn_cast<Attribute>()
                               .dyn_cast_or_null<ValueOrPoisonLikeAttr>();
    if (trueAttr && falseAttr) return bitSelect(condition, trueAttr, falseAttr);

    return {};
}

OpFoldResult BitFolder::bitSelect(
    OpFoldResult condition,
    OpFoldResult trueValue,
    OpFoldResult falseValue)
{
    assert(condition && trueValue && falseValue);

    // Handle trivial equality.
    if (trueValue == falseValue) return trueValue;

    // Fold if condition is constant.
    if (const auto condAttr = condition.dyn_cast<Attribute>()
                                  .dyn_cast_or_null<ValueOrPoisonLikeAttr>())
        return bitSelect(condAttr, trueValue, falseValue);

    // Fold if boolean pass-through.
    if (getType(trueValue).isSignlessInteger(1)) {
        const auto trueAttr =
            trueValue.dyn_cast<Attribute>().dyn_cast_or_null<BitSequenceAttr>();
        const auto falseAttr = falseValue.dyn_cast<Attribute>()
                                   .dyn_cast_or_null<BitSequenceAttr>();
        if (trueAttr && falseAttr && trueAttr.getValue().isOnes()
            && falseAttr.getValue().isZeros())
            return condition;
    }

    return {};
}

ValueOrPoisonLikeAttr
BitFolder::bitAnd(ValueOrPoisonLikeAttr lhs, ValueOrPoisonLikeAttr rhs)
{
    assert(lhs && rhs);
    assert(lhs.getType() == rhs.getType());

    // Zip the operands together.
    return zip(
        [](const auto &l, const auto &r) { return bitAnd(l, r); },
        lhs,
        rhs);
}

OpFoldResult BitFolder::bitAnd(OpFoldResult lhs, ValueOrPoisonLikeAttr rhs)
{
    assert(lhs && rhs);

    if (!rhs.isPoison()) {
        // Handle identities with 0 and 1.
        if (const auto splat = getSplat(rhs.getValueAttr())) {
            if (splat->isOnes()) return lhs;
            if (splat->isZeros()) return rhs;
        }
    }

    // Fold if all values are constant.
    if (const auto lhsAttr =
            lhs.dyn_cast<Attribute>().dyn_cast_or_null<ValueOrPoisonLikeAttr>())
        return bitAnd(lhsAttr, rhs);

    return {};
}

ValueOrPoisonLikeAttr
BitFolder::bitOr(ValueOrPoisonLikeAttr lhs, ValueOrPoisonLikeAttr rhs)
{
    assert(lhs && rhs);
    assert(lhs.getType() == rhs.getType());

    return zip(
        [](const auto &l, const auto &r) { return bitOr(l, r); },
        lhs,
        rhs);
}

OpFoldResult BitFolder::bitOr(OpFoldResult lhs, ValueOrPoisonLikeAttr rhs)
{
    assert(lhs && rhs);

    if (!rhs.isPoison()) {
        // Handle identities with 0 and 1.
        if (const auto splat = getSplat(rhs.getValueAttr())) {
            if (splat->isOnes()) return rhs;
            if (splat->isZeros()) return lhs;
        }
    }

    // Fold if all values are constant.
    if (const auto lhsAttr =
            lhs.dyn_cast<Attribute>().dyn_cast_or_null<ValueOrPoisonLikeAttr>())
        return bitOr(lhsAttr, rhs);

    return {};
}

ValueOrPoisonLikeAttr
BitFolder::bitXor(ValueOrPoisonLikeAttr lhs, ValueOrPoisonLikeAttr rhs)
{
    assert(lhs && rhs);
    assert(lhs.getType() == rhs.getType());

    return zip(
        [](const auto &l, const auto &r) { return bitXor(l, r); },
        lhs,
        rhs);
}

OpFoldResult BitFolder::bitXor(OpFoldResult lhs, ValueOrPoisonLikeAttr rhs)
{
    assert(lhs && rhs);

    if (!rhs.isPoison()) {
        // Handle identity with 0.
        if (const auto splat = getSplat(rhs.getValueAttr())) {
            if (splat->isZeros()) return lhs;
        }
    } else {
        // Handle poisoned operand.
        return rhs;
    }

    // Fold if all values are constant.
    if (const auto lhsAttr =
            lhs.dyn_cast<Attribute>().dyn_cast_or_null<ValueOrPoisonLikeAttr>())
        return bitXor(lhsAttr, rhs);

    return {};
}

ValueOrPoisonLikeAttr BitFolder::bitShl(
    ValueOrPoisonLikeAttr value,
    bit_width_t amount,
    ValueOrPoisonLikeAttr funnel)
{
    assert(value);
    assert(!funnel || (value.getType() == funnel.getType()));
    const auto elTy = value.getType().getElementType();
    const auto bitWidth = elTy.getBitWidth();

    if (!funnel) {
        // Perform normal shift.
        return map(
            [&](const auto &el) { return bitShl(bitWidth, el, amount); },
            value);
    }

    // Perform funnel shift.
    return zip(
        [&](const auto &v, const auto &f) {
            return bitShl(bitWidth, v, f, amount);
        },
        value,
        funnel);
}

OpFoldResult
BitFolder::bitShl(OpFoldResult value, bit_width_t amount, OpFoldResult funnel)
{
    assert(value);
    const auto valueTy = getType(value).cast<BitSequenceLikeType>();
    const auto bitWidth = valueTy.getElementType().getBitWidth();

    // Handle neutral case.
    if (amount == 0) return value;

    // Handle shift out case.
    if ((amount >= (2 * bitWidth)) || (!funnel && (amount >= bitWidth)))
        return ValueLikeAttr::getSplat(valueTy, Const::zeros(bitWidth));

    const auto funnelAttr = funnel
                                ? funnel.dyn_cast<Attribute>()
                                      .dyn_cast_or_null<ValueOrPoisonLikeAttr>()
                                : ValueOrPoisonLikeAttr{};
    const auto valueAttr =
        value.dyn_cast<Attribute>().dyn_cast_or_null<ValueOrPoisonLikeAttr>();

    // Handle dynamic funnel.
    if (funnel && !funnelAttr) {
        if (amount < bitWidth && valueAttr && valueAttr.isPoison())
            return valueAttr;
        return {};
    }

    // Handle poisoned funnel.
    if (funnelAttr && funnelAttr.isPoison()) return funnelAttr;

    // Fold if all values are constant.
    if (valueAttr) return bitShl(valueAttr, amount, funnelAttr);

    return {};
}

OpFoldResult BitFolder::bitShl(
    OpFoldResult value,
    ub::ValueOrPoisonAttr<IntegerAttr> amount,
    OpFoldResult funnel)
{
    assert(value && amount);
    assert(amount.getType().isIndex());

    // Handle poisoned amount.
    if (amount.isPoison()) return ub::PoisonAttr::get(getType(value));

    return bitShl(
        value,
        static_cast<bit_width_t>(
            amount.getValueAttr().getValue().getZExtValue()),
        funnel);
}

ValueOrPoisonLikeAttr BitFolder::bitShr(
    ValueOrPoisonLikeAttr value,
    bit_width_t amount,
    ValueOrPoisonLikeAttr funnel)
{
    assert(value);
    assert(!funnel || (value.getType() == funnel.getType()));
    const auto elTy = value.getType().getElementType();
    const auto bitWidth = elTy.getBitWidth();

    if (!funnel) {
        // Perform normal shift.
        return map(
            [&](const auto &el) { return bitShr(bitWidth, el, amount); },
            value);
    }

    // Perform funnel shift.
    return zip(
        [&](const auto &v, const auto &f) {
            return bitShr(bitWidth, v, f, amount);
        },
        value,
        funnel);
}

OpFoldResult
BitFolder::bitShr(OpFoldResult value, bit_width_t amount, OpFoldResult funnel)
{
    assert(value);
    const auto valueTy = getType(value).cast<BitSequenceLikeType>();
    const auto bitWidth = valueTy.getElementType().getBitWidth();

    // Handle neutral case.
    if (amount == 0) return value;

    // Handle shift out case.
    if ((amount >= (2 * bitWidth)) || (!funnel && (amount >= bitWidth)))
        return ValueLikeAttr::getSplat(valueTy, Const::zeros(bitWidth));

    const auto funnelAttr = funnel
                                ? funnel.dyn_cast<Attribute>()
                                      .dyn_cast_or_null<ValueOrPoisonLikeAttr>()
                                : ValueOrPoisonLikeAttr{};
    const auto valueAttr =
        value.dyn_cast<Attribute>().dyn_cast_or_null<ValueOrPoisonLikeAttr>();

    // Handle dynamic funnel.
    if (funnel && !funnelAttr) {
        if (amount < bitWidth && valueAttr && valueAttr.isPoison())
            return valueAttr;
        return {};
    }

    // Handle poisoned funnel.
    if (funnelAttr && funnelAttr.isPoison()) return funnelAttr;

    // Fold if all values are constant.
    if (valueAttr) return bitShr(valueAttr, amount, funnelAttr);

    return {};
}

OpFoldResult BitFolder::bitShr(
    OpFoldResult value,
    ub::ValueOrPoisonAttr<IntegerAttr> amount,
    OpFoldResult funnel)
{
    assert(value && amount);
    assert(amount.getType().isIndex());

    // Handle poisoned amount.
    if (amount.isPoison()) return ub::PoisonAttr::get(getType(value));

    return bitShr(
        value,
        static_cast<bit_width_t>(
            amount.getValueAttr().getValue().getZExtValue()),
        funnel);
}
