/// Declares the Bit dialect enums.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

#include <compare>
#include <cstdint>
#include <optional>
#include <type_traits>

//===- Generated includes -------------------------------------------------===//

#include "base2-mlir/Dialect/Bit/Enums.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::bit {

//===----------------------------------------------------------------------===//
// EqualityPredicate
//===----------------------------------------------------------------------===//

/// Determines whether @p equal is matched by @p pred .
[[nodiscard]] constexpr bool matches(bool equal, EqualityPredicate pred)
{
    using underlying_type = std::underlying_type_t<EqualityPredicate>;
    const auto predBits = static_cast<underlying_type>(pred);

    // NOTE: This uses the defined bit structure.
    return predBits & (0b10 >> equal);
}

/// Negates @p pred .
[[nodiscard]] constexpr EqualityPredicate negate(EqualityPredicate pred)
{
    using underlying_type = std::underlying_type_t<EqualityPredicate>;
    auto predBits = static_cast<underlying_type>(pred);

    // NOTE: This uses the defined bit structure.
    predBits = (~predBits) & 0b11;

    return static_cast<EqualityPredicate>(predBits);
}

} // namespace mlir::bit
