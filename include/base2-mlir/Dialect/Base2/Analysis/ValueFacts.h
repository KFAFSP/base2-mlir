/// Declares the ValueFacts enumeration and friends.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "base2-mlir/Dialect/Base2/Interfaces/BitSequenceAttr.h"
#include "base2-mlir/Dialect/Base2/Interfaces/BitSequenceType.h"

#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::base2 {

//===----------------------------------------------------------------------===//
// ValueFacts
//===----------------------------------------------------------------------===//

/// Enumeration of bitflags that encode known facts about a number value.
enum class ValueFacts {
    /// Nothing is known.
    None = 0b00000000,
    /// The value is known to be positive.
    Positive = 0b00000001,
    /// The value is known to be negative.
    Negative = 0b00000010,
    /// The magnitude is known to be zero.
    Zero = 0b00000100,
    /// The magnitude is known to be one.
    One = 0b00001000,
    /// The value is known to be the minimum representable.
    Min = 0b00010000,
    /// The value is known to be the maximum representable.
    Max = 0b00100000,
    /// The magntiude is known to be infinite.
    Inf = 0b01000000,
    /// The value is known to not represent a number.
    NaN = 0b10000000,
    LLVM_MARK_AS_BITMASK_ENUM(NaN),

    /// All facts (useful as a mask).
    All = 0b11111111
};

LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

/// Tests whether @p facts satisfies the @p conjunction .
[[nodiscard]] constexpr bool all(ValueFacts facts, ValueFacts conjunction)
{
    return (facts & conjunction) == conjunction;
}
/// Tests whether @p facts satisfies the @p disjunction .
[[nodiscard]] constexpr bool any(ValueFacts facts, ValueFacts disjunction)
{
    return (facts & disjunction) != ValueFacts::None;
}

/// Writes flags in @p facts to @p out .
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &out, ValueFacts facts)
{
    llvm::SmallVector<StringRef> flags;
    if (all(facts, ValueFacts::Positive)) flags.push_back("+");
    if (all(facts, ValueFacts::Negative)) flags.push_back("-");
    if (all(facts, ValueFacts::Zero)) flags.push_back("0");
    if (all(facts, ValueFacts::One)) flags.push_back("1");
    if (all(facts, ValueFacts::Min)) flags.push_back("min");
    if (all(facts, ValueFacts::Max)) flags.push_back("max");
    if (all(facts, ValueFacts::Inf)) flags.push_back("inf");
    if (all(facts, ValueFacts::NaN)) flags.push_back("NaN");

    out << "{";
    llvm::interleaveComma(flags, out);
    return out << "}";
}

//===----------------------------------------------------------------------===//
// Signum
//===----------------------------------------------------------------------===//

/// Enumeration of signum values.
enum class Signum : int { Negative = -1, Zero = 0, Positive = 1 };

/// Gets the Signum of @p facts , if known.
[[nodiscard]] constexpr std::optional<Signum> signum(ValueFacts facts)
{
    if (all(facts, ValueFacts::Zero)) return Signum::Zero;
    if (all(facts, ValueFacts::Negative)) return Signum::Negative;
    if (all(facts, ValueFacts::Positive)) return Signum::Positive;
    return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Magnitude
//===----------------------------------------------------------------------===//

/// Enumeration of special magnitudes.
enum class Magnitude : int { Zero = 0, One, Infinite };

/// Gets the Magnitude of @p facts , if known.
[[nodiscard]] constexpr std::optional<Magnitude> magnitude(ValueFacts facts)
{
    if (all(facts, ValueFacts::Zero)) return Magnitude::Zero;
    if (all(facts, ValueFacts::One)) return Magnitude::One;
    if (all(facts, ValueFacts::Inf)) return Magnitude::Infinite;
    return std::nullopt;
}

} // namespace mlir::base2
