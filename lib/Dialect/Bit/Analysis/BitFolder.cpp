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

    // Quick exit for verum and falsum.
    const auto i1Ty = IntegerType::get(lhs.getContext(), 1);
    const auto resultTy = lhs.getType().getSameShape(i1Ty);
    switch (predicate) {
    case EqualityPredicate::Verum:
        return BitSequenceLikeAttr::get(resultTy, true);
    case EqualityPredicate::Falsum:
        return BitSequenceLikeAttr::get(resultTy, false);
    default: break;
    }

    // Zip the operands together.
    return zip(
        [=](const auto &lhs, const auto &rhs) -> ConstOrPoison {
            if (!lhs || !rhs) return poison;
            return matches(*lhs == *rhs, predicate);
        },
        lhs,
        rhs,
        i1Ty);
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
    case EqualityPredicate::Verum:
        return BitSequenceLikeAttr::getSplat(outTy, true);
    case EqualityPredicate::Falsum:
        return BitSequenceLikeAttr::getSplat(outTy, false);
    default: break;
    }

    // Handle poisoned case.
    if (ub::isPoison(lhs) || ub::isPoison(rhs))
        return ValueOrPoisonLikeAttr::get(outTy);

    // Handle trivial case.
    if (lhs == rhs)
        return BitSequenceLikeAttr::getSplat(outTy, matches(true, predicate));

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
    assert(succeeded(
        verifyCompatibleShape(trueValue.getType(), falseValue.getType())));

    // Handle poisoned condition.
    if (condition.isPoison())
        return ValueOrPoisonLikeAttr::get(trueValue.getType());

    if (const auto cond = condition.dyn_cast<ValueAttr>()) {
        // Quick exit if the condition is just a bool.
        return cond.getValue().isOnes() ? trueValue : falseValue;
    }

    assert(succeeded(
        verifyCompatibleShape(condition.getType(), trueValue.getType())));

    // Perform a three-way zip.
    return zip(
        [](const auto &c, const auto &t, const auto &f) -> ConstOrPoison {
            if (!c) return poison;
            return c->isOnes() ? t : f;
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

    if (const auto cond = condition.dyn_cast<ValueAttr>()) {
        // Quick exit if the condition is just a bool.
        return cond.getValue().isOnes() ? trueValue : falseValue;
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

    return {};
}

ValueLikeAttr BitFolder::bitCmpl(ValueLikeAttr value)
{
    assert(value);

    return value.map([](const auto &val) { return val.logicCmpl(); });
}

ValueLikeAttr BitFolder::bitAnd(ValueLikeAttr lhs, ValueLikeAttr rhs)
{
    assert(lhs && rhs);
    assert(lhs.getType() == rhs.getType());

    return lhs.zip(
        [](const auto &lhs, const auto &rhs) { return lhs.logicAnd(rhs); },
        rhs);
}

OpFoldResult BitFolder::bitAnd(OpFoldResult lhs, ValueLikeAttr rhs)
{
    assert(lhs && rhs);

    if (const auto attr = rhs.dyn_cast<ValueAttr>()) {
        if (attr.getValue().isOnes()) return lhs;
        if (attr.getValue().isZeros()) return rhs;
    }

    if (const auto attr =
            lhs.dyn_cast<Attribute>().dyn_cast_or_null<ValueLikeAttr>())
        return bitAnd(attr, rhs);

    return OpFoldResult{};
}

ValueLikeAttr BitFolder::bitOr(ValueLikeAttr lhs, ValueLikeAttr rhs)
{
    assert(lhs && rhs);
    assert(lhs.getType() == rhs.getType());

    return lhs.zip(
        [](const auto &lhs, const auto &rhs) { return lhs.logicOr(rhs); },
        rhs);
}

OpFoldResult BitFolder::bitOr(OpFoldResult lhs, ValueLikeAttr rhs)
{
    assert(lhs && rhs);

    if (const auto attr = rhs.dyn_cast<ValueAttr>()) {
        if (attr.getValue().isZeros()) return lhs;
        if (attr.getValue().isOnes()) return rhs;
    }

    if (const auto attr =
            lhs.dyn_cast<Attribute>().dyn_cast_or_null<ValueLikeAttr>())
        return bitOr(attr, rhs);

    return OpFoldResult{};
}

ValueLikeAttr BitFolder::bitXor(ValueLikeAttr lhs, ValueLikeAttr rhs)
{
    assert(lhs && rhs);
    assert(lhs.getType() == rhs.getType());

    return lhs.zip(
        [](const auto &lhs, const auto &rhs) { return lhs.logicXor(rhs); },
        rhs);
}

OpFoldResult BitFolder::bitXor(OpFoldResult lhs, ValueLikeAttr rhs)
{
    assert(lhs && rhs);

    if (const auto attr = rhs.dyn_cast<ValueAttr>()) {
        if (attr.getValue().isZeros()) return lhs;
    }

    if (const auto attr =
            lhs.dyn_cast<Attribute>().dyn_cast_or_null<ValueLikeAttr>())
        return bitXor(attr, rhs);

    return OpFoldResult{};
}

ValueLikeAttr
BitFolder::bitShl(ValueLikeAttr value, bit_width_t amount, ValueLikeAttr funnel)
{
    assert(value);
    assert(!funnel || (value.getType() == funnel.getType()));
    const auto elTy = value.getType().getElementType();
    const auto bitWidth = elTy.getBitWidth();

    if (amount == 0) return value;
    if (amount >= 2 * bitWidth || (!funnel && amount >= bitWidth))
        return ValueLikeAttr::getSplat(
            value.getType(),
            BitSequence::zeros(bitWidth));

    if (!funnel)
        return value.map(
            [&](const BitSequence &v) { return v.logicShl(amount); });

    return value.zip(
        [&](BitSequence v, BitSequence f) { return v.funnelShl(f, amount); },
        funnel);
}

OpFoldResult
BitFolder::bitShl(OpFoldResult value, bit_width_t amount, OpFoldResult funnel)
{
    assert(value);
    const auto attr =
        value.dyn_cast<Attribute>().dyn_cast_or_null<ValueLikeAttr>();
    const auto valueTy =
        attr ? attr.getType()
             : value.dyn_cast<Value>().getType().cast<BitSequenceLikeType>();
    const auto elTy = valueTy.getElementType();
    const auto bitWidth = elTy.getBitWidth();

    if (amount == 0) return value;
    if (amount >= 2 * bitWidth || (!funnel && amount >= bitWidth))
        return ValueLikeAttr::getSplat(valueTy, BitSequence::zeros(bitWidth));

    if (!attr) return OpFoldResult{};
    if (!funnel) return bitShl(attr, amount);

    const auto funnelAttr =
        funnel.dyn_cast<Attribute>().dyn_cast_or_null<ValueLikeAttr>();
    if (!funnelAttr) return OpFoldResult{};

    return bitShl(attr, amount, funnelAttr);
}

ValueLikeAttr
BitFolder::bitShr(ValueLikeAttr value, bit_width_t amount, ValueLikeAttr funnel)
{
    assert(value);
    assert(!funnel || (value.getType() == funnel.getType()));
    const auto elTy = value.getType().getElementType();
    const auto bitWidth = elTy.getBitWidth();

    if (amount == 0) return value;
    if (amount >= 2 * bitWidth || (!funnel && amount >= bitWidth))
        return ValueLikeAttr::getSplat(
            value.getType(),
            BitSequence::zeros(bitWidth));

    if (!funnel)
        return value.map(
            [&](const BitSequence &v) { return v.logicShr(amount); });

    return value.zip(
        [&](BitSequence v, BitSequence f) { return v.funnelShr(f, amount); },
        funnel);
}

OpFoldResult
BitFolder::bitShr(OpFoldResult value, bit_width_t amount, OpFoldResult funnel)
{
    assert(value);
    const auto attr =
        value.dyn_cast<Attribute>().dyn_cast_or_null<ValueLikeAttr>();
    const auto valueTy =
        attr ? attr.getType()
             : value.dyn_cast<Value>().getType().cast<BitSequenceLikeType>();
    const auto elTy = valueTy.getElementType();
    const auto bitWidth = elTy.getBitWidth();

    if (amount == 0) return value;
    if (amount >= 2 * bitWidth || (!funnel && amount >= bitWidth))
        return ValueLikeAttr::getSplat(valueTy, BitSequence::zeros(bitWidth));

    if (!attr) return OpFoldResult{};
    if (!funnel) return bitShr(attr, amount);

    const auto funnelAttr =
        funnel.dyn_cast<Attribute>().dyn_cast_or_null<ValueLikeAttr>();
    if (!funnelAttr) return OpFoldResult{};

    return bitShr(attr, amount, funnelAttr);
}
