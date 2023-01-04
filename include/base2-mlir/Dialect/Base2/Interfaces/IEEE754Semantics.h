/// Declares the Base2 IEEE754Semantics interface.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "base2-mlir/Dialect/Base2/Analysis/BitSequence.h"
#include "base2-mlir/Dialect/Base2/Enums.h"
#include "base2-mlir/Dialect/Base2/Interfaces/BitSequenceType.h"
#include "base2-mlir/Dialect/Base2/Interfaces/InterpretableType.h"
#include "mlir/IR/OpDefinition.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <bit>
#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>

namespace mlir::base2 {

/// Type that holds an IEEE-754 number's exponent value.
using exponent_t = std::int32_t;
/// Maximum number of bits in an IEEE-754 type.
static constexpr auto max_exponent_bits =
    std::numeric_limits<exponent_t>::digits;
/// Smallest possible exponent_t value.
static constexpr auto min_exponent = std::numeric_limits<exponent_t>::min();
/// Largest possible exponent_t value.
static constexpr auto max_exponent = std::numeric_limits<exponent_t>::max();

/// Gets the number of bits required to encode @p value .
[[nodiscard]] constexpr auto getBitWidth(exponent_t value)
{
    return std::bit_width(static_cast<std::uint32_t>(value));
}

/// Calculate the minimum exponent value for @p bias .
[[nodiscard]] constexpr exponent_t getMinExponent(exponent_t bias)
{
    return exponent_t(1) - bias;
}

/// Calculate the maximum exponent value for @p bias and @p exponentBits .
///
/// @pre    `exponentBits >= 2 && exponentBits <= max_exponent_bits`
[[nodiscard]] constexpr exponent_t
getMaxExponent(exponent_t bias, bit_width_t exponentBits)
{
    assert(exponentBits >= 2 && exponentBits <= max_exponent_bits);

    const auto biasedMax = (std::int64_t(1) << exponentBits) - 1;
    return (biasedMax - 1) - bias;
}

/// Calculate the distance between @p maxExponent and @p minExponent .
///
/// @pre    `minExponent <= maxExponent`
[[nodiscard]] constexpr std::uint64_t
getExponentRange(exponent_t maxExponent, exponent_t minExponent)
{
    assert(minExponent <= maxExponent);

    return static_cast<std::uint64_t>(
        static_cast<std::int64_t>(maxExponent)
        - static_cast<std::int64_t>(minExponent));
}

/// Stores parameters for an IEEE-754 exponent (bit width and bias).
using exponent_params = std::pair<bit_width_t, exponent_t>;

/// Calculate the bit width and bias for the given exponent parameters.
///
/// @pre    `minExponent <= maxExponent`
[[nodiscard]] constexpr exponent_params
getExponentParams(exponent_t maxExponent, exponent_t minExponent)
{
    // Calculate the size of exponent range to avoid overflow
    const auto exponentRange = getExponentRange(maxExponent, minExponent);
    // Calculate the bias required to store the exponent values.
    const auto bias = exponent_t(1) - minExponent;
    // The biased exponent also needs to encode denormalized numbers (0...0)
    // and NaN/Inf (1...1).
    const auto biasedExponentRange = exponentRange + 2UL;
    // Calculate the number of bits needed to store the biased exponent.
    const auto exponentBits = std::bit_width(biasedExponentRange);

    return {exponentBits, bias};
}

/// Calculate the midpoint value for @p exponentBits .
///
/// @pre    `exponentBits >= 1 && exponentBits <= max_exponent_bits`
[[nodiscard]] constexpr exponent_t getBias(bit_width_t exponentBits)
{
    assert(exponentBits >= 1 && exponentBits <= max_exponent_bits);

    return (std::int64_t(1) << (exponentBits - 1)) - 1;
}

} // namespace mlir::base2

namespace mlir::base2::ieee754_semantics_interface_defaults {

/// Calculates the bias using the midpoint in the exponent field.
[[nodiscard]] exponent_t getBias(Type self);

} // namespace mlir::base2::ieee754_semantics_interface_defaults

//===- Generated includes -------------------------------------------------===//

#include "base2-mlir/Dialect/Base2/Interfaces/IEEE754Semantics.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::base2 {

//===----------------------------------------------------------------------===//
// IEEE754LikeType
//===----------------------------------------------------------------------===//

/// Concept for a type or container type of IEEE-754 numbers.
///
/// Satisfied by a BitSequenceLikeType type, the elements of which also satisfy
/// IEEE754Semantics.
class IEEE754LikeType : public BitSequenceLikeType {
public:
    /// @copydoc classof(Type)
    [[nodiscard]] static bool classof(BitSequenceLikeType type)
    {
        return type.getElementType().isa<IEEE754Semantics>();
    }
    /// Determines whether @p type is a IEEE754LikeType.
    [[nodiscard]] static bool classof(Type type)
    {
        if (const auto bitSequenceLikeTy = type.dyn_cast<BitSequenceLikeType>())
            return classof(bitSequenceLikeTy);

        return false;
    }

    using BitSequenceLikeType::BitSequenceLikeType;
    /*implicit*/ IEEE754LikeType(BitSequenceType) = delete;
    /// Initializes a IEEE754LikeType from @p type .
    ///
    /// @pre    `type`
    /*implicit*/ IEEE754LikeType(IEEE754Semantics type)
            : BitSequenceLikeType(type.cast<Type>().getImpl())
    {}

    /// Gets the underlying IEEE754Semantics.
    [[nodiscard]] IEEE754Semantics getSemantics() const
    {
        if (const auto self = dyn_cast<IEEE754Semantics>()) return self;

        return getElementType().cast<IEEE754Semantics>();
    }
};

/// Implements the IEEE754Semantics interface for built-in types.
void registerIEEE754SemanticsModels(MLIRContext &ctx);

} // namespace mlir::base2
