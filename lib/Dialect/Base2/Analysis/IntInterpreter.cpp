/// Implements the constant folding interpreter for llvm::APInt.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "base2-mlir/Dialect/Base2/Analysis/IntInterpreter.h"

using namespace mlir;
using namespace mlir::base2;

//===----------------------------------------------------------------------===//
// Checked arithmetic
//===----------------------------------------------------------------------===//

ValueFacts
IntInterpreter::getFacts(Signedness signedness, const llvm::APInt &value)
{
    auto facts = ValueFacts::None;

    if (value.isZero()) facts |= ValueFacts::Zero;
    if (signedness == Signedness::Signless) return facts;

    if (value.isOne()) facts |= ValueFacts::One;

    // Handle signed numbers.
    if (signedness == Signedness::Signed) {
        if (value.isAllOnes()) facts |= ValueFacts::One;
        if (value.isNegative()) {
            facts |= ValueFacts::Negative;
            if (value.isMinSignedValue()) facts |= ValueFacts::Min;
        } else {
            facts |= ValueFacts::Positive;
            if (value.isMaxSignedValue()) facts |= ValueFacts::Max;
        }
        return facts;
    }

    // Handle unsigned numbers.
    assert(signedness == Signedness::Unsigned);
    facts |= ValueFacts::Positive;
    if (value.isMinValue())
        facts |= ValueFacts::Min;
    else if (value.isMaxValue())
        facts |= ValueFacts::Max;
    return facts;
}

IntInterpreter::checked_result IntInterpreter::addChecked(
    bool isSigned,
    const llvm::APInt &lhs,
    const llvm::APInt &rhs)
{
    assert(lhs.getBitWidth() == rhs.getBitWidth());

    bool isInexact;
    const auto moduloResult =
        isSigned ? lhs.sadd_ov(rhs, isInexact) : lhs.uadd_ov(rhs, isInexact);
    const auto determineOverflow = [&]() {
        if (!isInexact) return Overflow::None;
        // On unsigned, only overflow can occur on add.
        if (!isSigned) return Overflow::Overflow;
        // If the sign changed to negative, an overflow occured.
        if (moduloResult.isNegative()) return Overflow::Overflow;
        // If the sign changed to positive, an underflow occured.
        return Overflow::Underflow;
    };

    return {determineOverflow(), moduloResult};
}

IntInterpreter::checked_result IntInterpreter::subChecked(
    bool isSigned,
    const llvm::APInt &lhs,
    const llvm::APInt &rhs)
{
    assert(lhs.getBitWidth() == rhs.getBitWidth());

    bool isInexact;
    const auto moduloResult =
        isSigned ? lhs.ssub_ov(rhs, isInexact) : lhs.usub_ov(rhs, isInexact);
    const auto determineOverflow = [&]() {
        if (!isInexact) return Overflow::None;
        // On unsigned, only underflow can occur on subtract.
        if (!isSigned) return Overflow::Underflow;
        // If the sign changed to negative, an overflow occured.
        if (moduloResult.isNegative()) return Overflow::Overflow;
        // If the sign changed to positive, an underflow occured.
        return Overflow::Underflow;
    };

    return {determineOverflow(), moduloResult};
}

IntInterpreter::checked_result IntInterpreter::mulChecked(
    bool isSigned,
    const llvm::APInt &lhs,
    const llvm::APInt &rhs)
{
    assert(lhs.getBitWidth() == rhs.getBitWidth());

    bool isInexact;
    const auto moduloResult =
        isSigned ? lhs.smul_ov(rhs, isInexact) : lhs.umul_ov(rhs, isInexact);
    const auto determineOverflow = [&]() {
        if (!isInexact) return Overflow::None;
        // On unsigned, only overflow can occur.
        if (!isSigned) return Overflow::Overflow;
        // On signed, equal signs always lead to overflow.
        if (lhs.isNegative() == rhs.isNegative()) return Overflow::Overflow;
        // Otherwise, it must be underflow.
        return Overflow::Underflow;
    };

    return {determineOverflow(), moduloResult};
}

IntInterpreter::checked_result IntInterpreter::divChecked(
    bool isSigned,
    const llvm::APInt &lhs,
    const llvm::APInt &rhs)
{
    assert(lhs.getBitWidth() == rhs.getBitWidth());

    auto isInexact = false;
    const auto moduloResult =
        isSigned ? lhs.sdiv_ov(rhs, isInexact) : lhs.udiv(rhs);
    const auto determineOverflow = [&]() {
        if (!isInexact) return Overflow::None;
        // Only overflow can occur (min / -1).
        return Overflow::Overflow;
    };

    return {determineOverflow(), moduloResult};
}

IntInterpreter::checked_result IntInterpreter::shlChecked(
    bool isSigned,
    const llvm::APInt &value,
    unsigned count)
{
    const llvm::APInt amount(64, count, false);

    bool isInexact;
    const auto moduloResult = isSigned ? value.sshl_ov(amount, isInexact)
                                       : value.ushl_ov(amount, isInexact);
    const auto determineOverflow = [&]() {
        if (!isInexact) return Overflow::None;
        // Direction depends on the sign.
        if (isSigned && value.isNegative()) return Overflow::Underflow;
        return Overflow::Overflow;
    };

    return {determineOverflow(), moduloResult};
}

IntInterpreter::checked_result IntInterpreter::truncChecked(
    bool isSigned,
    const llvm::APInt &value,
    unsigned outBitWidth)
{
    const auto isExact =
        isSigned ? value.isSignedIntN(outBitWidth) : value.isIntN(outBitWidth);
    const auto moduloResult = value.trunc(outBitWidth);
    const auto determineOverflow = [&]() {
        if (isExact) return Overflow::None;
        // Direction depends on the sign.
        if (isSigned && value.isNegative()) return Overflow::Underflow;
        return Overflow::Overflow;
    };

    return {determineOverflow(), moduloResult};
}

IntInterpreter::checked_result IntInterpreter::castChecked(
    bool isSigned,
    const llvm::APInt &value,
    bool outIsSigned,
    unsigned outBitWidth)
{
    // If a negative number is supposed to turn unsigned, we underflow.
    if (isSigned && !outIsSigned && value.isNegative())
        return {Overflow::Underflow, value.sextOrTrunc(outBitWidth)};
    // If an unsigned number turns negative, overflow occurs.
    if (!isSigned && outIsSigned && value.isNegative())
        return {Overflow::Overflow, value.sextOrTrunc(outBitWidth)};

    // If extension is performed, the result is always exact.
    if (outBitWidth >= value.getBitWidth())
        return isSigned ? value.sext(outBitWidth) : value.zext(outBitWidth);

    // We know the signedness does not cause overflow, so we can delegate to
    // regular truncation.
    return truncChecked(isSigned, value, outBitWidth);
}
