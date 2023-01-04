/// Implements the constant folding interpreter for fixed-point numbers.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "base2-mlir/Dialect/Base2/Analysis/FixedPointInterpreter.h"

using namespace mlir;
using namespace mlir::base2;

//===----------------------------------------------------------------------===//
// Casting
//===----------------------------------------------------------------------===//

/// Calculates the bias value to add on rescaling.
///
/// @pre    `fractionalBits <= value.getBitWidth()`
static llvm::APInt getRescaleBias(
    bool isSigned,
    bit_width_t fractionalBits,
    APInt value,
    RoundingMode roundingMode)
{
    assert(fractionalBits <= value.getBitWidth());

    APInt bias(value.getBitWidth(), 0, false);

    switch (roundingMode) {
    case RoundingMode::None:
    case RoundingMode::RoundDown:
    case RoundingMode::Converge:
        // If nothing is done, the bits are truncated, and the value is
        // rounded down.
        break;

    case RoundingMode::Nearest:
    {
        // To round to nearest, ties to larger absolute value, we add 1 to
        // the first fractional bit.
        bias = APInt::getOneBitSet(value.getBitWidth(), fractionalBits - 1);
        // For negative values, this would round 0.5 up, so we subtract
        // the epsilon on negative to remedy this.
        bias -= (isSigned && value.isNegative());
        break;
    }
    case RoundingMode::RoundUp:
    {
        // To round up, we add the largest value smaller than 1.
        bias = APInt::getAllOnes(fractionalBits).zext(value.getBitWidth());
        break;
    }
    case RoundingMode::TowardsZero:
    {
        // To round towards zero, negative values must be rounded up.
        if (isSigned && value.isNegative())
            bias = APInt::getAllOnes(fractionalBits).zext(value.getBitWidth());
        break;
    }
    case RoundingMode::AwayFromZero:
    {
        // To round away from zero, positive values must be rounded up.
        if (!isSigned || !value.isNegative())
            bias = APInt::getAllOnes(fractionalBits).zext(value.getBitWidth());
        break;
    }
    }

    return bias;
}

FixedPointInterpreter::auto_result FixedPointInterpreter::rescale(
    FixedPointSemantics sema,
    const BitSequence &value,
    bit_width_t outFracBits,
    RoundingMode roundingMode)
{
    assert(sema);
    assert(value.size() == sema.getBitWidth());
    if (sema.isSignless()) return std::nullopt;

    // Initialize the result value.
    auto result = value.asUInt();

    // Determine the relative scaling of out to input.
    const auto relativeScale = static_cast<long>(sema.getFractionalBits())
                               - static_cast<long>(outFracBits);

    // Handle no-op case.
    if (relativeScale == 0) return {sema, result};

    // Prepare the new semantics.
    const auto resultSema = FixedPointSemantics::get(
        sema.getContext(),
        sema.getSignedness(),
        sema.getIntegerBits(),
        outFracBits);

    // Handle extension case.
    if (relativeScale > 0) {
        // Extend the result to fit additional fractional bits.
        const auto resultBits = result.getBitWidth() + relativeScale;
        result =
            sema.isSigned() ? result.sext(resultBits) : result.zext(resultBits);

        // Shift left to match the original scale.
        result <<= relativeScale;
        return {resultSema, result};
    }

    // Calculate the rescaling bias value.
    const auto bias = getRescaleBias(
        sema.isSigned(),
        sema.getFractionalBits(),
        result,
        roundingMode);

    // Add the rescaling bias with rounding.
    result = IntInterpreter::add(sema.isSigned(), result, bias, roundingMode)
                 .getValue();

    // Perform the right shift.
    if (sema.isSigned())
        result.ashrInPlace(-relativeScale);
    else
        result.lshrInPlace(-relativeScale);
    return {resultSema, result};
}

bit_result FixedPointInterpreter::valueCast(
    FixedPointSemantics from,
    const BitSequence &value,
    FixedPointSemantics to,
    RoundingMode roundingMode)
{
    assert(from && to);
    assert(value.size() == from.getBitWidth());
    if (from.isSignless() || to.isSignless()) return std::nullopt;

    // Initialize the result value.
    auto result = value.asUInt();

    // If the result type has more integer bits, we need to promote first.
    if (to.getIntegerBits() > from.getIntegerBits()) {
        const auto interBits = to.getIntegerBits() + from.getFractionalBits();
        result =
            from.isSigned() ? result.sext(interBits) : result.zext(interBits);
        from = FixedPointSemantics::get(
            from.getContext(),
            from.getSignedness(),
            to.getIntegerBits(),
            from.getFractionalBits());
    }

    // Rescale to match the result scale.
    result = rescale(from, result, to.getFractionalBits(), roundingMode)
                 .getValue()
                 .asUInt();

    // Perform the integer cast.
    return IntInterpreter::cast(
               from.isSigned(),
               result,
               to.isSigned(),
               to.getBitWidth(),
               roundingMode)
        .getValue();
}

//===----------------------------------------------------------------------===//
// Exact operations
//===----------------------------------------------------------------------===//

/// Determines the supertype semantics of @p lhs and @p rhs .
///
/// @pre    `lhs`
/// @pre    `rhs`
[[nodiscard]] static std::tuple<Signedness, bit_width_t, bit_width_t>
align(FixedPointSemantics lhs, FixedPointSemantics rhs)
{
    assert(lhs && rhs);

    // Take the maximum of all widths.
    auto outIntBits = std::max(lhs.getIntegerBits(), rhs.getIntegerBits());
    const auto outFracBits =
        std::max(lhs.getFractionalBits(), rhs.getFractionalBits());

    // Unify signedness.
    const auto signedness = super(lhs.getSignedness(), rhs.getSignedness());
    if (signedness == Signedness::Signed) {
        // When converting unsigned to signed, an additional bit is needed.
        if (lhs.isUnsigned())
            outIntBits = std::max(outIntBits, lhs.getIntegerBits() + 1);
        else if (rhs.isUnsigned())
            outIntBits = std::max(outIntBits, rhs.getIntegerBits() + 1);
    }

    return {signedness, outIntBits, outFracBits};
}

/// Extends @p in to @p outBits .
///
/// @pre    `outBits >= in.getBitWidth()`
static llvm::APInt
extend(bool isSigned, const llvm::APInt &in, bit_width_t outBits)
{
    assert(outBits >= in.getBitWidth());

    return isSigned ? in.sext(outBits) : in.zext(outBits);
}

/// Aligns @p in to @p outFractionalBits , assuming it is wide enough.
///
/// @pre    `inFractionalBits <= outFractionalBits`
/// @pre    no overflow will occur
static llvm::APInt alignFractionalExact(
    bool isSigned,
    bit_width_t inFractionalBits,
    const llvm::APInt in,
    bit_width_t outFractionalBits)
{
    assert(inFractionalBits <= outFractionalBits);

    if (inFractionalBits < outFractionalBits) {
        const auto checked = IntInterpreter::shlChecked(
            isSigned,
            in,
            outFractionalBits - inFractionalBits);
        assert(checked.isExact());
        return checked.getValue();
    }

    return in;
}

/// Adjusts @p value with @p from semantics to the supertype @p to .
///
/// @pre    `from`
/// @pre    `to`
/// @pre    `value.size() == from.getBitWidth()`
/// @pre    `!from.isSignless()`
/// @pre    `to.isSupersetOf(from)`
/// @pre    `!to.isSignless()`
BitSequence alignExact(
    FixedPointSemantics from,
    const BitSequence &value,
    FixedPointSemantics to)
{
    assert(from && to);
    assert(value.size() == from.getBitWidth());
    assert(to.isSupersetOf(from));
    assert(!from.isSignless() && !to.isSignless());

    // Extent ot match the output semantics.
    auto result = extend(from.isSigned(), value.asUInt(), to.getBitWidth());

    // Align the fractional parts.
    result = alignFractionalExact(
        to.isSigned(),
        from.getFractionalBits(),
        result,
        to.getFractionalBits());

    return result;
}

/// Adjusts @p lhs and @p rhs to the common supertype @p outSema .
///
/// @pre    `lhsSema`
/// @pre    `rhsSema`
/// @pre    `lhs.size() == lhsSema.getBitWidth()`
/// @pre    `rhs.size() == rhsSema.getBitWidth()`
/// @pre    `!lhsSema.isSignless() && !rhsSema.isSignless()`
/// @pre    `outSema`
/// @pre    `!outSema.isSignless()`
/// @pre    `outSema.isSupersetOf(lhsSema) && outSema.isSupersetOf(rhsSema)`
FixedPointInterpreter::align_result alignExact(
    FixedPointSemantics lhsSema,
    const BitSequence &lhs,
    FixedPointSemantics rhsSema,
    const BitSequence &rhs,
    FixedPointSemantics outSema)
{
    return {
        outSema,
        alignExact(lhsSema, lhs.asUInt(), outSema),
        alignExact(rhsSema, rhs.asUInt(), outSema)};
}

FixedPointSemantics
FixedPointInterpreter::align(FixedPointSemantics lhs, FixedPointSemantics rhs)
{
    const auto [signedness, outIntBits, outFracBits] = ::align(lhs, rhs);
    if (signedness == Signedness::Signless) return FixedPointSemantics{};

    return FixedPointSemantics::get(
        lhs.getContext(),
        signedness,
        outIntBits,
        outFracBits);
}

FixedPointInterpreter::align_result FixedPointInterpreter::align(
    FixedPointSemantics lhsSema,
    const BitSequence &lhs,
    FixedPointSemantics rhsSema,
    const BitSequence &rhs)
{
    if (const auto outSema = align(lhsSema, rhsSema))
        return ::alignExact(lhsSema, lhs, rhsSema, rhs, outSema);

    return std::nullopt;
}

FixedPointSemantics
FixedPointInterpreter::add(FixedPointSemantics lhs, FixedPointSemantics rhs)
{
    // Start from aligned semantics.
    auto [signedness, outIntBits, outFracBits] = ::align(lhs, rhs);
    if (signedness == Signedness::Signless) return FixedPointSemantics{};

    // Addition can produce 1 additional bit at maximum.
    ++outIntBits;

    return FixedPointSemantics::get(
        lhs.getContext(),
        signedness,
        outIntBits,
        outFracBits);
}

FixedPointInterpreter::auto_result FixedPointInterpreter::add(
    FixedPointSemantics lhsSema,
    const BitSequence &lhs,
    FixedPointSemantics rhsSema,
    const BitSequence &rhs)
{
    const auto outSema = add(lhsSema, rhsSema);
    if (!outSema) return std::nullopt;

    // Align to the exact semantics.
    const auto [_, lhsExt, rhsExt] =
        ::alignExact(lhsSema, lhs, rhsSema, rhs, outSema);

    // Perform operation, which must be exact.
    const auto checked = IntInterpreter::add(
        outSema.isSigned(),
        lhsExt.asUInt(),
        rhsExt.asUInt());
    assert(checked.isExact());
    return {outSema, checked.getValue()};
}

FixedPointSemantics
FixedPointInterpreter::sub(FixedPointSemantics lhs, FixedPointSemantics rhs)
{
    // Start from aligned semantics.
    auto [signedness, outIntBits, outFracBits] = ::align(lhs, rhs);
    if (signedness == Signedness::Signless) return FixedPointSemantics{};

    // Subtraction can produce 1 additional bit.
    ++outIntBits;

    if (signedness == Signedness::Unsigned) {
        // When subtracting unsigned numbers, the result must be signed.
        signedness = Signedness::Signed;
    }

    return FixedPointSemantics::get(
        lhs.getContext(),
        signedness,
        outIntBits,
        outFracBits);
}

FixedPointInterpreter::auto_result FixedPointInterpreter::sub(
    FixedPointSemantics lhsSema,
    const BitSequence &lhs,
    FixedPointSemantics rhsSema,
    const BitSequence &rhs)
{
    const auto outSema = add(lhsSema, rhsSema);
    if (!outSema) return std::nullopt;

    // Align to the exact semantics.
    const auto [_, lhsExt, rhsExt] =
        ::alignExact(lhsSema, lhs, rhsSema, rhs, outSema);

    // Perform operation, which must be exact.
    const auto checked = IntInterpreter::sub(
        outSema.isSigned(),
        lhsExt.asUInt(),
        rhsExt.asUInt());
    assert(checked.isExact());
    return {outSema, checked.getValue()};
}

FixedPointSemantics
FixedPointInterpreter::mul(FixedPointSemantics lhs, FixedPointSemantics rhs)
{
    assert(lhs && rhs);

    // The bit widths and fractional bits add.
    auto outBits = lhs.getBitWidth() + rhs.getBitWidth();
    const auto outFracBits = lhs.getFractionalBits() + rhs.getFractionalBits();

    const auto signedness = super(lhs.getSignedness(), rhs.getSignedness());
    if (signedness == Signedness::Signless) return FixedPointSemantics{};
    if (signedness == Signedness::Signed) {
        // If the signedness is opposing, or both operands are signed, 1
        // additional bit may be requiredd.
        ++outBits;
    }

    return FixedPointSemantics::get(
        lhs.getContext(),
        signedness,
        outBits - outFracBits,
        outFracBits);
}

FixedPointInterpreter::auto_result FixedPointInterpreter::mul(
    FixedPointSemantics lhsSema,
    const BitSequence &lhs,
    FixedPointSemantics rhsSema,
    const BitSequence &rhs)
{
    const auto outSema = mul(lhsSema, rhsSema);
    if (!outSema) return std::nullopt;

    // Extend the operands.
    const auto lhsExt =
        extend(lhsSema.isSigned(), lhs.asUInt(), outSema.getBitWidth());
    const auto rhsExt =
        extend(rhsSema.isSigned(), rhs.asUInt(), outSema.getBitWidth());

    // Perform operation, which must be exact.
    const auto checked =
        IntInterpreter::mulChecked(outSema.isSigned(), lhsExt, rhsExt);
    assert(checked.isExact());
    return {outSema, checked.getValue()};
}

FixedPointSemantics
FixedPointInterpreter::div(FixedPointSemantics lhs, FixedPointSemantics rhs)
{
    // Start from aligned semantics.
    auto [signedness, outIntBits, outFracBits] = ::align(lhs, rhs);
    if (signedness == Signedness::Signless) return FixedPointSemantics{};

    // We need space for the biggest integer and the pre-shift.
    auto outBits = outIntBits + 2 * outFracBits;
    if (signedness == Signedness::Signed) {
        // If the result is signed, we may need an exta bit to compute min/-1.
        outBits++;
    }

    return FixedPointSemantics::get(
        lhs.getContext(),
        signedness,
        outBits - outFracBits,
        outFracBits);
}

FixedPointInterpreter::auto_result FixedPointInterpreter::div(
    FixedPointSemantics lhsSema,
    const BitSequence &lhs,
    FixedPointSemantics rhsSema,
    const BitSequence &rhs)
{
    const auto outSema = div(lhsSema, rhsSema);
    if (!outSema) return std::nullopt;

    // Align to the exact semantics.
    const auto [_, lhsExt, rhsExt] =
        ::alignExact(lhsSema, lhs, rhsSema, rhs, outSema);

    // Pre-shift the dividend, which may not overflow.
    const auto checkedShl = IntInterpreter::shlChecked(
        outSema.isSigned(),
        lhsExt.asUInt(),
        outSema.getFractionalBits());
    assert(checkedShl.isExact());

    // Perform the division, which may not overflow.
    const auto checkedDiv = IntInterpreter::divChecked(
        outSema.isSigned(),
        checkedShl.getValue(),
        rhsExt.asUInt());
    assert(checkedDiv.isExact());
    return {outSema, checkedDiv.getValue()};
}
