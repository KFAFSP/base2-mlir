/// Implements the Bit dialect folding helper.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "base2-mlir/Dialect/Bit/Analysis/BitFolder.h"

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace mlir::bit;

ValueLikeAttr BitFolder::bitCast(ValueLikeAttr in, BitSequenceLikeType resultTy)
{
    assert(in && resultTy);
    assert(
        in.getElementType().getBitWidth()
        == resultTy.getElementType().getBitWidth());
    assert(succeeded(verifyCompatibleShape(in.getType(), resultTy)));

    return in.bitCastElements(resultTy.getElementType());
}

ValueLikeAttr BitFolder::bitCmp(
    EqualityPredicate predicate,
    ValueLikeAttr lhs,
    ValueLikeAttr rhs)
{
    assert(lhs && rhs);
    assert(
        lhs.getElementType().getBitWidth()
        == rhs.getElementType().getBitWidth());
    assert(succeeded(verifyCompatibleShape(lhs.getType(), rhs.getType())));

    // Quick exit for verum and falsum.
    const auto i1Ty = IntegerType::get(lhs.getContext(), 1);
    switch (predicate) {
    case EqualityPredicate::Verum: return ValueAttr::get(i1Ty, true);
    case EqualityPredicate::Falsum: return ValueAttr::get(i1Ty, false);
    default: break;
    }

    // Zip the operands together.
    return lhs.zip(
        [=](const auto &lhs, const auto &rhs) {
            return matches(lhs == rhs, predicate);
        },
        rhs,
        i1Ty);
}

ValueLikeAttr BitFolder::bitSelect(
    ValueLikeAttr condition,
    ValueLikeAttr trueValue,
    ValueLikeAttr falseValue)
{
    assert(condition && trueValue && falseValue);
    assert(condition.getElementType().isSignlessInteger(1));
    assert(trueValue.getElementType() == falseValue.getElementType());
    assert(succeeded(
        verifyCompatibleShape(trueValue.getType(), falseValue.getType())));

    if (const auto cond = condition.dyn_cast<ValueAttr>()) {
        // Quick exit if the condition is just a bool.
        return cond.getValue().isOnes() ? trueValue : falseValue;
    }

    assert(succeeded(
        verifyCompatibleShape(condition.getType(), trueValue.getType())));

    // Perform a three-way zip.
    const auto denseTrue = trueValue.cast<DenseBitSequencesAttr>();
    const auto denseFalse = falseValue.cast<DenseBitSequencesAttr>();
    if (denseTrue.isSplat()) {
        return condition.zip(
            [t = denseTrue.value_begin()](
                const auto &c,
                const auto &f) mutable { return c.isOnes() ? *t++ : f; },
            denseFalse,
            denseFalse.getElementType());
    }
    return condition.zip(
        [f = denseFalse.value_begin()](const auto &c, const auto &t) mutable {
            return c.isOnes() ? t : *f++;
        },
        denseTrue,
        denseTrue.getElementType());
}

OpFoldResult BitFolder::bitSelect(
    ValueLikeAttr condition,
    OpFoldResult trueValue,
    OpFoldResult falseValue)
{
    assert(condition);

    if (const auto cond = condition.dyn_cast<ValueAttr>()) {
        // Quick exit if the condition is just a bool.
        return cond.getValue().isOnes() ? trueValue : falseValue;
    }

    // Fold if all values are constant.
    const auto trueAttr =
        trueValue.dyn_cast<Attribute>().dyn_cast_or_null<ValueLikeAttr>();
    const auto falseAttr =
        falseValue.dyn_cast<Attribute>().dyn_cast_or_null<ValueLikeAttr>();
    if (trueAttr && falseAttr) return bitSelect(condition, trueAttr, falseAttr);

    return OpFoldResult{};
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
