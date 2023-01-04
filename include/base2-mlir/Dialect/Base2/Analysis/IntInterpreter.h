/// Declares the constant folding interpreter for llvm::APInt.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "base2-mlir/Dialect/Base2/Analysis/BitSequence.h"
#include "base2-mlir/Dialect/Base2/Analysis/ValueFacts.h"
#include "base2-mlir/Dialect/Base2/Enums.h"

#include "llvm/ADT/APInt.h"

#include <compare>
#include <utility>

namespace mlir::base2 {

/// Implements constant folding for llvm::APInt.
///
/// This extends the llvm::APInt operations to respect the RoundingMode
/// specified on the operation.
class IntInterpreter {
public:
    //===------------------------------------------------------------------===//
    // Comparison
    //===------------------------------------------------------------------===//

    /// Compares two llvm::APInt instances as unsigned integers.
    [[nodiscard]] static std::strong_ordering
    cmpUI(const llvm::APInt &lhs, const llvm::APInt &rhs)
    {
        const auto szCmp = lhs.getActiveWords() <=> rhs.getActiveWords();
        if (!std::is_eq(szCmp)) return szCmp;

        // They don't let us access llvm::APInt::compare ...

        return llvm::APInt::tcCompare(
                   lhs.getRawData(),
                   rhs.getRawData(),
                   lhs.getActiveWords())
               <=> 0;
    }
    /// Compares two llvm::APInt instances as signed integers.
    [[nodiscard]] static std::strong_ordering
    cmpSI(const llvm::APInt &lhs, const llvm::APInt &rhs)
    {
        // They don't let us access llvm::APInt::compareSigned ...

        const auto sgnCmp = lhs.isNegative() <=> rhs.isNegative();
        if (!std::is_eq(sgnCmp)) return 0 <=> sgnCmp;

        return cmpUI(lhs, rhs);
    }

    /// Compares two llvm::APInt instances.
    [[nodiscard]] static std::strong_ordering
    cmp(bool isSigned, const llvm::APInt &lhs, const llvm::APInt &rhs)
    {
        return isSigned ? cmpSI(lhs, rhs) : cmpUI(lhs, rhs);
    }

    /// Infers ValueFacts from @p value .
    [[nodiscard]] static ValueFacts
    getFacts(Signedness signedness, const llvm::APInt &value);

    //===------------------------------------------------------------------===//
    // Factories
    //===------------------------------------------------------------------===//

    /// Gets the zero value for @p width .
    static llvm::APInt getZero(bit_width_t width)
    {
        return llvm::APInt(width, 0, false);
    }
    /// Gets the minimum value for @p isSigned and @p width .
    static llvm::APInt getMin(bool isSigned, bit_width_t width)
    {
        return isSigned ? llvm::APInt::getSignedMinValue(width)
                        : llvm::APInt::getMinValue(width);
    }
    /// Gets the maximum value for @p isSigned and @p width .
    static llvm::APInt getMax(bool isSigned, bit_width_t width)
    {
        return isSigned ? llvm::APInt::getSignedMaxValue(width)
                        : llvm::APInt::getMaxValue(width);
    }

    //===------------------------------------------------------------------===//
    // Checked arithmetic
    //===------------------------------------------------------------------===//

    /// Enumeration of possible overflow types.
    enum class Overflow { None = 0, Overflow, Underflow };

    /// Result type for a checked arithmetic operation.
    struct [[nodiscard]] checked_result : std::pair<Overflow, llvm::APInt> {
        /// Initializes a checked_result from @p overflow and @p value .
        /*implicit*/ checked_result(Overflow overflow, llvm::APInt value)
                : pair(overflow, std::move(value))
        {}
        /// Initializes an exact checked_result from @p value .
        /*implicit*/ checked_result(llvm::APInt value)
                : checked_result(Overflow::None, value)
        {}

        /// Gets the type of IntOverflow.
        [[nodiscard]] Overflow getOverflow() const { return first; }
        /// Gets the underlying llvm::APInt value.
        [[nodiscard]] const llvm::APInt &getValue() const & { return second; }
        /// @copydoc getValue()
        [[nodiscard]] llvm::APInt &&getValue() && { return std::move(second); }

        /// Determines whether the result is indicated to be exact.
        [[nodiscard]] bool isExact() const
        {
            return getOverflow() == Overflow::None;
        }

        /// Applies @p fn to the overflow and value if not exact.
        checked_result handle(auto fn) &&
        {
            if (isExact()) return std::move(*this);
            return fn(getOverflow(), std::move(*this).getValue());
        }
        /// @copydoc handle()
        checked_result handle(auto fn) const &
        {
            if (isExact()) return *this;
            return fn(getOverflow(), getValue());
        }

        /// @copydoc getOverflow()
        [[nodiscard]] /*implicit*/ operator Overflow() const
        {
            return getOverflow();
        }
    };

    // Adds two llvm::APInt instances with overflow detection.
    //
    // @pre     `lhs.getBitWidth() == rhs.getBitWidth()`
    static checked_result
    addChecked(bool isSigned, const llvm::APInt &lhs, const llvm::APInt &rhs);
    // Subtracts two llvm::APInt instances with overflow detection.
    //
    // @pre     `lhs.getBitWidth() == rhs.getBitWidth()`
    static checked_result
    subChecked(bool isSigned, const llvm::APInt &lhs, const llvm::APInt &rhs);
    // Multiplies two llvm::APInt instances with overflow detection.
    //
    // @pre     `lhs.getBitWidth() == rhs.getBitWidth()`
    static checked_result
    mulChecked(bool isSigned, const llvm::APInt &lhs, const llvm::APInt &rhs);
    // Divides two llvm::APInt instances with overflow detection.
    //
    // @pre     `lhs.getBitWidth() == rhs.getBitWidth()`
    static checked_result
    divChecked(bool isSigned, const llvm::APInt &lhs, const llvm::APInt &rhs);

    /// Shifts @p value left by @p count with overflow detection.
    static checked_result
    shlChecked(bool isSigned, const llvm::APInt &value, unsigned count);

    /// Truncates @p value to @p outBitWidth with overflow detection.
    ///
    /// @pre    `outBitWidth <= value.getBitWidth()`
    static checked_result
    truncChecked(bool isSigned, const llvm::APInt &value, unsigned outBitWidth);

    /// Casts @p value to new signedness and bit width with overflow detection.
    static checked_result castChecked(
        bool isSigned,
        const llvm::APInt &value,
        bool outIsSigned,
        unsigned outBitWidth);

    //===------------------------------------------------------------------===//
    // Rounded arithmetic
    //===------------------------------------------------------------------===//

    /// Applies @p roundingMode to @p checked if it is inexact.
    static checked_result round(
        bool isSigned,
        checked_result checked,
        RoundingMode roundingMode = RoundingMode::None)
    {
        return std::move(checked).handle([=](Overflow overflow, auto &&value) {
            switch (overflow) {
            case Overflow::Underflow:
                if (saturateIntUnderflow(roundingMode))
                    return getMin(isSigned, value.getBitWidth());
                break;
            case Overflow::Overflow:
                if (saturateIntOverflow(roundingMode))
                    return getMax(isSigned, value.getBitWidth());
                break;

            default: llvm_unreachable("unknown Overflow");
            }

            return value;
        });
    }

    // Adds two llvm::APInt instances with rounding.
    //
    // @pre     `lhs.getBitWidth() == rhs.getBitWidth()`
    static checked_result
    add(bool isSigned,
        const llvm::APInt &lhs,
        const llvm::APInt &rhs,
        RoundingMode roundingMode = RoundingMode::None)
    {
        return round(isSigned, addChecked(isSigned, lhs, rhs), roundingMode);
    }
    // Subtracts two llvm::APInt instances with rounding.
    //
    // @pre     `lhs.getBitWidth() == rhs.getBitWidth()`
    static checked_result
    sub(bool isSigned,
        const llvm::APInt &lhs,
        const llvm::APInt &rhs,
        RoundingMode roundingMode = RoundingMode::None)
    {
        return round(isSigned, subChecked(isSigned, lhs, rhs), roundingMode);
    }
    // Multiplies two llvm::APInt instances with rounding.
    //
    // @pre     `lhs.getBitWidth() == rhs.getBitWidth()`
    static checked_result
    mul(bool isSigned,
        const llvm::APInt &lhs,
        const llvm::APInt &rhs,
        RoundingMode roundingMode = RoundingMode::None)
    {
        return round(isSigned, mulChecked(isSigned, lhs, rhs), roundingMode);
    }
    // Divides two llvm::APInt instances with rounding.
    //
    // @pre     `lhs.getBitWidth() == rhs.getBitWidth()`
    static checked_result
    div(bool isSigned,
        const llvm::APInt &lhs,
        const llvm::APInt &rhs,
        RoundingMode roundingMode = RoundingMode::None)
    {
        return round(isSigned, divChecked(isSigned, lhs, rhs), roundingMode);
    }

    /// Shifts @p value left by @p count with rounding.
    static checked_result
    shl(bool isSigned,
        const llvm::APInt &value,
        unsigned count,
        RoundingMode roundingMode = RoundingMode::None)
    {
        return round(
            isSigned,
            shlChecked(isSigned, value, count),
            roundingMode);
    }

    /// Truncates @p value to @p bitWidth with rounding.
    ///
    /// @pre    `outBitWidth <= value.getBitWidth()`
    static checked_result trunc(
        bool isSigned,
        const llvm::APInt &value,
        unsigned outBitWidth,
        RoundingMode roundingMode = RoundingMode::None)
    {
        return round(
            isSigned,
            truncChecked(isSigned, value, outBitWidth),
            roundingMode);
    }

    /// Casts @p value to new signedness and bit width with rounding.
    static checked_result cast(
        bool isSigned,
        const llvm::APInt &value,
        bool outIsSigned,
        unsigned outBitWidth,
        RoundingMode roundingMode = RoundingMode::None)
    {
        return round(
            outIsSigned,
            castChecked(isSigned, value, outIsSigned, outBitWidth),
            roundingMode);
    }
};

} // namespace mlir::base2
