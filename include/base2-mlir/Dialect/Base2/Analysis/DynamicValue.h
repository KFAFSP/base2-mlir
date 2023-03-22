/// Declares the DynamicValue type used during compile-time reasoning.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "base2-mlir/Dialect/Base2/Analysis/BitInterpreter.h"
#include "base2-mlir/Dialect/Base2/Analysis/ValueFacts.h"
#include "base2-mlir/Dialect/Bit/Analysis/BitSequence.h"
#include "mlir/IR/Value.h"

#include <compare>

namespace mlir::base2 {

//===----------------------------------------------------------------------===//
// DynamicValueBase
//===----------------------------------------------------------------------===//

/// Base class for a runtime or compile-time value with known ValueFacts.
class [[nodiscard]] DynamicValueBase {
public:
    /// Initializes a DynamicValueBase for an unknown value.
    /*implicit*/ constexpr DynamicValueBase() : m_facts(ValueFacts::None) {}
    /// Initializes a DynamicValueBase with @p facts .
    /*implicit*/ constexpr DynamicValueBase(ValueFacts facts) : m_facts(facts)
    {}

    /// Gets the ValueFacts known about the value.
    [[nodiscard]] constexpr ValueFacts getFacts() const { return m_facts; }
    /// @copydoc getFacts()
    [[nodiscard]] constexpr operator ValueFacts() const { return getFacts(); }

#define FACT_TEST(name)                                                        \
    [[nodiscard]] constexpr bool is##name() const                              \
    {                                                                          \
        return all(getFacts(), ValueFacts::name);                              \
    }

    FACT_TEST(Positive)
    FACT_TEST(Negative)
    FACT_TEST(Zero)
    FACT_TEST(One)
    FACT_TEST(Min)
    FACT_TEST(Max)
    FACT_TEST(Inf)
    [[nodiscard]] constexpr bool isPosInf() const
    {
        return all(getFacts(), ValueFacts::Positive | ValueFacts::Inf);
    }
    [[nodiscard]] constexpr bool isNegInf() const
    {
        return all(getFacts(), ValueFacts::Negative | ValueFacts::Inf);
    }
    FACT_TEST(NaN)

#undef FACT_TEST

    /// Gets the Signum if it is known.
    [[nodiscard]] constexpr std::optional<Signum> getSignum() const
    {
        return signum(getFacts());
    }
    /// Gets the Magnitude if it is known.
    [[nodiscard]] constexpr std::optional<Magnitude> getMagnitude() const
    {
        return magnitude(getFacts());
    }

    //===------------------------------------------------------------------===//
    // Common operations
    //===------------------------------------------------------------------===//

    /// Attempts to compare two DynamicValueBases.
    [[nodiscard]] cmp_result cmp(DynamicValueBase rhs) const;

protected:
    ValueFacts m_facts;
};

//===----------------------------------------------------------------------===//
// DynamicValue
//===----------------------------------------------------------------------===//

/// Holds a run-time or compile-time value with associated ValueFacts.
class [[nodiscard]] DynamicValue : public DynamicValueBase {
public:
    /// Initializes an unbound DynamicValue.
    /*implicit*/ DynamicValue() = default;
    /// Initializes a DynamicValue bound to @p value with @p facts .
    /*implicit*/ DynamicValue(Value value, ValueFacts facts = ValueFacts::None);
    /// Initializes a DynamicValue bound to @p attr .
    /*implicit*/ DynamicValue(bit::BitSequenceLikeAttr attr);

    /// Gets the OpFoldResult binding.
    [[nodiscard]] OpFoldResult getBinding() const { return m_binding; }
    /// @copydoc getBinding()
    [[nodiscard]] /*implicit*/ operator OpFoldResult() const
    {
        return getBinding();
    }
    /// Determines whether a value is bound.
    [[nodiscard]] bool isBound() const { return !getBinding().isNull(); }
    /// @copydoc isBound()
    [[nodiscard]] /*implicit*/ operator bool() const { return isBound(); }

    /// Gets the BitSequenceLikeType.
    ///
    /// @pre    `isBound()`
    [[nodiscard]] bit::BitSequenceLikeType getType() const
    {
        if (const auto variable = getVariable())
            return variable.getType().cast<bit::BitSequenceLikeType>();

        return getConstant().getType();
    }
    /// Gets the underlying BitSequenceType.
    ///
    /// @pre    `isBound()`
    [[nodiscard]] bit::BitSequenceType getElementType() const
    {
        return getType().getElementType();
    }

    /// Gets the bound Value, if any.
    [[nodiscard]] Value getVariable() const
    {
        return getBinding().dyn_cast<Value>();
    }
    /// Gets the bound BitSequenceAttr, if any.
    [[nodiscard]] bit::BitSequenceLikeAttr getConstant() const
    {
        if (const auto attr = getBinding().dyn_cast<Attribute>())
            return attr.cast<bit::BitSequenceLikeAttr>();

        return bit::BitSequenceLikeAttr{};
    }

    //===------------------------------------------------------------------===//
    // Common operations
    //===------------------------------------------------------------------===//

    /// Determines the result of comparing two DynamicValues.
    DynamicValue cmp(PartialOrderingPredicate pred, DynamicValue rhs) const;

    /// Determines the mimimum of two DynamicValues.
    DynamicValue min(DynamicValue rhs) const;
    /// Determines the maximum of two DynamicValues.
    DynamicValue max(DynamicValue rhs) const;

    //===------------------------------------------------------------------===//
    // Closed arithmetic operations
    //===------------------------------------------------------------------===//

    /// Determines the sum of two DynamicValues.
    DynamicValue
    add(DynamicValue rhs, RoundingMode roundingMode = RoundingMode::None) const;
    /// Determines the difference of two DynamicValues.
    DynamicValue
    sub(DynamicValue rhs, RoundingMode roundingMode = RoundingMode::None) const;
    /// Determines the product of two DynamicValues.
    DynamicValue
    mul(DynamicValue rhs, RoundingMode roundingMode = RoundingMode::None) const;
    /// Determines the quotient of two DynamicValues.
    DynamicValue
    div(DynamicValue rhs, RoundingMode roundingMode = RoundingMode::None) const;
    /// Determines the remainder of dividing two DynamicValues.
    DynamicValue mod(DynamicValue rhs) const;

private:
    OpFoldResult m_binding;
};

} // namespace mlir::base2
