/// Implements the DynamicValue type used during compile-time reasoning.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "base2-mlir/Dialect/Base2/Analysis/DynamicValue.h"

#include "base2-mlir/Dialect/Base2/IR/Base2.h"
#include "base2-mlir/Dialect/Bit/IR/Bit.h"

#define DEBUG_TYPE "base2-dynamicvalue"

using namespace mlir;
using namespace mlir::base2;

//===----------------------------------------------------------------------===//
// DynamicValueBase
//===----------------------------------------------------------------------===//

cmp_result DynamicValueBase::cmp(DynamicValueBase rhs) const
{
    // NaNs are not ordered.
    if (isNaN() || rhs.isNaN()) return std::partial_ordering::unordered;

    const auto lhsSign = getSignum();
    const auto rhsSign = rhs.getSignum();
    // Values of unknown signum cannot be ordered.
    if (!lhsSign || !rhsSign) return std::nullopt;

    // Values of different signs are trivially ordered.
    if (*lhsSign < *rhsSign) return std::partial_ordering::less;
    if (*lhsSign > *rhsSign) return std::partial_ordering::greater;

    // Infinite and non-infinity are trivially ordered.
    if (isInf() ^ rhs.isInf()) {
        const auto absCmp = isInf() <=> rhs.isInf();
        return isPositive() ? absCmp : 0 <=> absCmp;
    }

    const auto lhsMag = getMagnitude();
    const auto rhsMag = rhs.getMagnitude();
    // Only definite values can be compared.
    if (!lhsMag || !rhsMag) return std::nullopt;

    // Make a surrogate that behaves the same under comparison.
    const auto lhsInt = static_cast<int>(*lhsSign) * static_cast<int>(*lhsMag);
    const auto rhsInt = static_cast<int>(*rhsSign) * static_cast<int>(*rhsMag);
    return lhsInt <=> rhsInt;
}

//===----------------------------------------------------------------------===//
// DynamicValue
//===----------------------------------------------------------------------===//

DynamicValue::DynamicValue(Value value, ValueFacts facts)
        : DynamicValueBase(facts),
          m_binding(value)
{
    if (!value) return;

    // Look through ConstantOp results.
    if (auto constOp = value.getDefiningOp<bit::ConstantOp>()) {
        const auto attr = constOp.getValue();
        m_binding = attr;
        m_facts |= BitInterpreter::getFacts(attr);
    }
}

DynamicValue::DynamicValue(bit::BitSequenceLikeAttr attr)
        : DynamicValueBase(BitInterpreter::getFacts(attr)),
          m_binding(attr)
{}

DynamicValue
DynamicValue::cmp(PartialOrderingPredicate pred, DynamicValue rhs) const
{
    // Try constant comparison.
    if (const auto result =
            BitInterpreter::cmp(pred, getConstant(), rhs.getConstant()))
        return result;

    // Try fact-based comparison.
    if (const auto ordering = DynamicValueBase::cmp(rhs)) {
        const auto makeSplat = [&](bool value) -> Attribute {
            if (const auto shapedTy = getType().dyn_cast<ShapedType>())
                return DenseIntElementsAttr::get(shapedTy, value);
            return BoolAttr::get(getType().getContext(), value);
        };

        return makeSplat(matches(*ordering, pred))
            .cast<bit::BitSequenceLikeAttr>();
    }

    return DynamicValue{};
}

DynamicValue DynamicValue::min(DynamicValue rhs) const
{
    // NaNs are treated as missing data.
    if (isNaN()) return rhs;
    if (rhs.isNaN()) return *this;

    // Try constant folding.
    if (const auto result =
            BitInterpreter::min(getConstant(), rhs.getConstant()))
        return result;

    // Try fact-based comparison.
    const auto factBased =
        cmp(PartialOrderingPredicate::OrderedAndLessOrEqual, rhs);
    if (factBased.isOne()) return *this;
    if (factBased.isZero()) return rhs;

    // Try trivial equality.
    if (getVariable() == rhs.getVariable()) return *this;

    // We know nothing.
    return DynamicValue{};
}

DynamicValue DynamicValue::max(DynamicValue rhs) const
{
    // NaNs are treated as missing data.
    if (isNaN()) return rhs;
    if (rhs.isNaN()) return *this;

    // Try constant folding.
    if (const auto result =
            BitInterpreter::max(getConstant(), rhs.getConstant()))
        return result;

    // Try fact-based comparison.
    const auto factBased =
        cmp(PartialOrderingPredicate::OrderedAndGreaterOrEqual, rhs);
    if (factBased.isOne()) return *this;
    if (factBased.isZero()) return rhs;

    // Try trivial equality.
    if (getVariable() == rhs.getVariable()) return *this;

    // We know nothing.
    return DynamicValue{};
}

DynamicValue
DynamicValue::add(DynamicValue rhs, RoundingMode roundingMode) const
{
    // NaNs propagate.
    if (isNaN()) return *this;
    if (rhs.isNaN()) return rhs;

    // Neutral element.
    if (isZero()) return rhs;
    if (rhs.isZero()) return *this;

    // Try constant folding.
    return BitInterpreter::add(getConstant(), rhs.getConstant(), roundingMode);
}

DynamicValue
DynamicValue::sub(DynamicValue rhs, RoundingMode roundingMode) const
{
    // NaNs propagate.
    if (isNaN()) return *this;
    if (rhs.isNaN()) return rhs;

    // Neutral element.
    if (rhs.isZero()) return *this;

    // Try constant folding.
    return BitInterpreter::sub(getConstant(), rhs.getConstant(), roundingMode);
}

DynamicValue
DynamicValue::mul(DynamicValue rhs, RoundingMode roundingMode) const
{
    // NaNs propagate.
    if (isNaN()) return *this;
    if (rhs.isNaN()) return rhs;

    // Neutral element.
    if (isOne()) return rhs;
    if (rhs.isOne()) return *this;

    // Try constant folding.
    return BitInterpreter::mul(getConstant(), rhs.getConstant(), roundingMode);
}

DynamicValue
DynamicValue::div(DynamicValue rhs, RoundingMode roundingMode) const
{
    // NaNs propagate.
    if (isNaN()) return *this;
    if (rhs.isNaN()) return rhs;

    // Neutral element.
    if (rhs.isOne() && rhs.isPositive()) return *this;

    // Try constant folding.
    return BitInterpreter::div(getConstant(), rhs.getConstant(), roundingMode);
}

DynamicValue DynamicValue::mod(DynamicValue rhs) const
{
    // NaNs propagate.
    if (isNaN()) return *this;
    if (rhs.isNaN()) return rhs;

    // Try constant folding.
    return BitInterpreter::mod(getConstant(), rhs.getConstant());
}
