/// Declares the Base2 dialect enums.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/FloatingPointMode.h"

#include <compare>
#include <cstdint>
#include <optional>
#include <type_traits>

namespace mlir::arith {

enum class CmpFPredicate : std::uint64_t;
enum class CmpIPredicate : std::uint64_t;

} // namespace mlir::arith

//===- Generated includes -------------------------------------------------===//

#include "base2-mlir/Dialect/Base2/Enums.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::base2 {

//===----------------------------------------------------------------------===//
// Signedness
//===----------------------------------------------------------------------===//

/// Converts a @p builtinSignedness to a Signedness value.
[[nodiscard]] constexpr Signedness
getSignedness(IntegerType::SignednessSemantics builtinSignedness)
{
    switch (builtinSignedness) {
    case IntegerType::SignednessSemantics::Signless:
        return Signedness::Signless;
    case IntegerType::SignednessSemantics::Signed: return Signedness::Signed;
    case IntegerType::SignednessSemantics::Unsigned:
        return Signedness::Unsigned;
    }
}
/// Converts a @p signedness to a IntegerType::SignednessSemantics value.
[[nodiscard]] constexpr IntegerType::SignednessSemantics
getBuiltinSignedness(Signedness signedness)
{
    switch (signedness) {
    case Signedness::Signless:
        return IntegerType::SignednessSemantics::Signless;
    case Signedness::Signed: return IntegerType::SignednessSemantics::Signed;
    case Signedness::Unsigned:
        return IntegerType::SignednessSemantics::Unsigned;
    }
}

/// Obtains the Signedness of the supertype of @p lhs and @p rhs .
///
/// The supertype of two types with signdeness @p lhs and @p rhs is guaranteed
/// to be able to exactly represent any value of the two subtypes. Thus, it
/// must unify their signedness:
///
/// - If any of @p lhs or @p rhs is signless, the result is signless.
/// - If any of @p lhs or @p rhs is signed, the result is signed.
/// - Otherwise, the result is unsigned.
///
/// Since signless numbers do not have any semantics, they work like a poison
/// value that indicates implementation-specific behavior.
[[nodiscard]] constexpr Signedness super(Signedness lhs, Signedness rhs)
{
    if (lhs == Signedness::Signless || rhs == Signedness::Signless)
        return Signedness::Signless;
    if (lhs == Signedness::Signed || rhs == Signedness::Signed)
        return Signedness::Signed;
    return Signedness::Unsigned;
}

//===----------------------------------------------------------------------===//
// PartialOrderingPredicate
//===----------------------------------------------------------------------===//

/// Determines whether @p mask is matched by @p pred .
[[nodiscard]] constexpr bool
matches(std::partial_ordering mask, PartialOrderingPredicate pred)
{
    using underlying_type = std::underlying_type_t<PartialOrderingPredicate>;
    const auto predBits = static_cast<underlying_type>(pred);

    // NOTE: This uses the defined bit structure.
    if (std::is_eq(mask)) return (predBits & 0b0001);
    if (std::is_gt(mask)) return (predBits & 0b0010);
    if (std::is_lt(mask)) return (predBits & 0b0100);
    return (predBits & 0b1000);
}

/// Flips the relational part of @p pred .
///
/// Swaps the greater for less and vice versa, without touching the equal, less
/// ordered, or unordered predicate.
[[nodiscard]] constexpr PartialOrderingPredicate
flip(PartialOrderingPredicate pred)
{
    using underlying_type = std::underlying_type_t<PartialOrderingPredicate>;
    auto predBits = static_cast<underlying_type>(pred);

    // NOTE: This uses the defined bit structure.
    const underlying_type gt = predBits & 0b0010;
    const underlying_type lt = predBits & 0b0100;
    predBits ^= gt ^ lt;
    predBits ^= (gt << 1) ^ (lt >> 1);

    return static_cast<PartialOrderingPredicate>(predBits);
}
/// Negates @p pred .
///
/// Obtains the exact inverse of the predicate, including swapping ordered with
/// unordered and vice versea.
[[nodiscard]] constexpr PartialOrderingPredicate
negate(PartialOrderingPredicate pred)
{
    using underlying_type = std::underlying_type_t<PartialOrderingPredicate>;
    auto predBits = static_cast<underlying_type>(pred);

    // NOTE: This uses the defined bit structure.
    predBits = (~predBits) & 0b1111;

    return static_cast<PartialOrderingPredicate>(predBits);
}
/// Obtains the strongly-ordered version of @p pred .
///
/// - If @p pred is Ordered, the result is Verum.
/// - If @p pred is Unordered, the result is Falsum.
/// - Otherwise, the unordered predicate is removed.
[[nodiscard]] constexpr PartialOrderingPredicate
strong(PartialOrderingPredicate pred)
{
    if (pred == PartialOrderingPredicate::Ordered)
        return PartialOrderingPredicate::Verum;

    using underlying_type = std::underlying_type_t<PartialOrderingPredicate>;
    auto predBits = static_cast<underlying_type>(pred);

    // NOTE: This uses the defined bit structure.
    predBits &= 0b0111;

    return static_cast<PartialOrderingPredicate>(predBits);
}

/// Converts @p pred to an arith::CmpFPredicate.
[[nodiscard]] arith::CmpFPredicate
getCmpFPredicate(PartialOrderingPredicate pred);
/// Converts @p pred to an arith::CmpIPredicate, if possible.
[[nodiscard]] std::optional<arith::CmpIPredicate>
getCmpIPredicate(Signedness signedness, PartialOrderingPredicate pred);

//===----------------------------------------------------------------------===//
// RoundingMode
//===----------------------------------------------------------------------===//

/// Converts @p roundingMode to an llvm::RoundingMode value.
///
/// The result may be llvm::RoundingMode::Invalid if not supported.
[[nodiscard]] constexpr llvm::RoundingMode
getLLVMRoundingMode(RoundingMode roundingMode)
{
    switch (roundingMode) {
    case RoundingMode::None: return llvm::RoundingMode::TowardZero;
    case RoundingMode::Nearest: return llvm::RoundingMode::NearestTiesToAway;
    case RoundingMode::RoundUp: return llvm::RoundingMode::TowardPositive;
    case RoundingMode::RoundDown: return llvm::RoundingMode::TowardNegative;
    case RoundingMode::TowardsZero: return llvm::RoundingMode::TowardZero;
    default: return llvm::RoundingMode::Invalid;
    }
}

/// Determines whether @p roundingMode saturates on integer overflow.
[[nodiscard]] constexpr bool saturateIntOverflow(RoundingMode roundingMode)
{
    using underlying_type = std::underlying_type_t<RoundingMode>;
    const auto modeBits = static_cast<underlying_type>(roundingMode);

    return modeBits & 0b1000;
}
/// Determines whether @p roundingMode saturates on integer underflow.
[[nodiscard]] constexpr bool saturateIntUnderflow(RoundingMode roundingMode)
{
    using underlying_type = std::underlying_type_t<RoundingMode>;
    const auto modeBits = static_cast<underlying_type>(roundingMode);

    return modeBits & 0b0100;
}

} // namespace mlir::base2
