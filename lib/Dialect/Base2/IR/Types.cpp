/// Implements the Base2 dialect types.
///
/// @file
/// @author     Jihaong Bi (jiahong.bi@mailbox.tu-dresden.de)
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "base2-mlir/Dialect/Base2/IR/Types.h"

#include "base2-mlir/Dialect/Base2/Analysis/FixedPointInterpreter.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/APFixedPoint.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "base2-types"

using namespace mlir;
using namespace mlir::base2;

//===- Generated implementation -------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "base2-mlir/Dialect/Base2/IR/Types.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// FixedPointType
//===----------------------------------------------------------------------===//

LogicalResult FixedPointType::verify(
    function_ref<InFlightDiagnostic()> emitError,
    IntegerType integerType,
    bit::bit_width_t fractionalBits)
{
    if (!integerType) return emitError() << "expected IntegerType";

    // Fractional bits must fit into integer type.
    if (fractionalBits > integerType.getWidth())
        return emitError()
               << "number of fractional bits (" << fractionalBits
               << ") exceeds bit width (" << integerType.getWidth() << ")";

    return success();
}

LogicalResult FixedPointType::verify(
    function_ref<InFlightDiagnostic()> emitError,
    Signedness,
    bit::bit_width_t integerBits,
    bit::bit_width_t fractionalBits)
{
    // Total bit width must fit into a standard integer type.
    const auto bitWidth = integerBits + fractionalBits;
    if (bitWidth > IntegerType::kMaxWidth)
        return emitError()
               << "total bits (" << bitWidth << ") exceed the limit ("
               << IntegerType::kMaxWidth << ")";

    return success();
}

void FixedPointType::print(AsmPrinter &printer) const
{
    // `<`
    printer << "<";

    // (`signed` | `unsigned`)?
    switch (getSignedness()) {
    case Signedness::Signless: break;
    case Signedness::Signed: printer << "signed "; break;
    case Signedness::Unsigned: printer << "unsigned "; break;
    }

    // $integerBits
    printer << getIntegerBits();
    // (`,` $fractionalBits)?
    if (getFractionalBits() > 0) printer << ", " << getFractionalBits();

    // `>`
    printer << ">";
}

Type FixedPointType::parse(AsmParser &parser)
{
    // `<`
    if (parser.parseLess()) return Type{};

    // (`signed` | `unsigned`)?
    auto signedness = Signedness::Signless;
    if (!parser.parseOptionalKeyword("signed"))
        signedness = Signedness::Signed;
    else if (!parser.parseOptionalKeyword("unsigned"))
        signedness = Signedness::Unsigned;

    // $integerBits
    unsigned integerBits, fractionalBits = 0;
    if (parser.parseInteger(integerBits)) return Type{};
    // (`,` $fractionalBits)?
    if (!parser.parseOptionalComma()) {
        if (parser.parseInteger(fractionalBits)) return Type{};
    }

    // `>`
    if (parser.parseGreater()) return Type{};

    // Custom verification for this representation.
    const auto emitError = [&]() {
        return parser.emitError(parser.getNameLoc());
    };
    if (failed(verify(emitError, signedness, integerBits, fractionalBits)))
        return Type{};

    return get(parser.getContext(), signedness, integerBits, fractionalBits);
}

[[nodiscard]] static bit_result valueCastTo(
    FixedPointSemantics from,
    const bit::BitSequence &value,
    InterpretableType to,
    RoundingMode roundingMode)
{
    if (from.isSignless()) return std::nullopt;

    if (const auto toFixed = to.dyn_cast<FixedPointSemantics>()) {
        return FixedPointInterpreter::valueCast(
            from,
            value,
            toFixed,
            roundingMode);
    }

    if (auto toFloat = to.dyn_cast<FloatType>()) {
        if (roundingMode != RoundingMode::None) return std::nullopt;

        llvm::APFixedPoint self(
            value.asUInt(),
            llvm::FixedPointSemantics(
                from.getBitWidth(),
                from.getFractionalBits(),
                from.isSigned(),
                false,
                false));

        return self.convertToFloat(toFloat.getFloatSemantics());
    }

    return std::nullopt;
}

[[nodiscard]] static bit_result valueCastFrom(
    InterpretableType from,
    const bit::BitSequence &value,
    FixedPointSemantics to,
    RoundingMode roundingMode)
{
    if (to.isSignless()) return std::nullopt;

    if (auto fromFloat = from.dyn_cast<FloatType>()) {
        if (roundingMode != RoundingMode::None) return std::nullopt;

        llvm::APFloat fltVal(fromFloat.getFloatSemantics(), value.asUInt());
        return llvm::APFixedPoint::getFromFloatValue(
                   fltVal,
                   llvm::FixedPointSemantics(
                       to.getBitWidth(),
                       to.getFractionalBits(),
                       to.isSigned(),
                       false,
                       false))
            .getValue();
    }

    return std::nullopt;
}

bit_result FixedPointType::valueCast(
    InterpretableType from,
    const bit::BitSequence &value,
    InterpretableType to,
    RoundingMode roundingMode) const
{
    const auto fromImpl = from.dyn_cast<FixedPointSemantics>();
    const auto toImpl = to.dyn_cast<FixedPointSemantics>();

    if (fromImpl) return valueCastTo(fromImpl, value, to, roundingMode);
    if (toImpl) return valueCastFrom(from, value, toImpl, roundingMode);

    return std::nullopt;
}

cmp_result FixedPointType::cmp(
    const bit::BitSequence &lhs,
    const bit::BitSequence &rhs) const
{
    return FixedPointInterpreter::cmp(*this, lhs, rhs);
}

bit_result FixedPointType::add(
    const bit::BitSequence &lhs,
    const bit::BitSequence &rhs,
    RoundingMode roundingMode) const
{
    return FixedPointInterpreter::add(*this, lhs, rhs, roundingMode);
}

bit_result FixedPointType::sub(
    const bit::BitSequence &lhs,
    const bit::BitSequence &rhs,
    RoundingMode roundingMode) const
{
    return FixedPointInterpreter::sub(*this, lhs, rhs, roundingMode);
}

bit_result FixedPointType::mul(
    const bit::BitSequence &lhs,
    const bit::BitSequence &rhs,
    RoundingMode roundingMode) const
{
    return FixedPointInterpreter::mul(*this, lhs, rhs, roundingMode);
}

bit_result FixedPointType::div(
    const bit::BitSequence &lhs,
    const bit::BitSequence &rhs,
    RoundingMode roundingMode) const
{
    return FixedPointInterpreter::div(*this, lhs, rhs, roundingMode);
}

bit_result FixedPointType::mod(
    const bit::BitSequence &lhs,
    const bit::BitSequence &rhs) const
{
    return FixedPointInterpreter::mod(*this, lhs, rhs);
}

ValueFacts FixedPointType::getFacts(const bit::BitSequence &value) const
{
    return FixedPointInterpreter::getFacts(*this, value);
}

//===----------------------------------------------------------------------===//
// IEEE754Type
//===----------------------------------------------------------------------===//

LogicalResult IEEE754Type::verify(
    function_ref<InFlightDiagnostic()> emitError,
    bit::bit_width_t precision,
    bit::bit_width_t exponentBits,
    exponent_t bias)
{
    // Precision must be within limits.
    if (precision < 2)
        return emitError()
               << "precision (" << precision << ") must be greater than 1";

    // Exponent bits must be within limits.
    if (exponentBits < 1 || exponentBits > max_exponent_bits)
        return emitError()
               << "exponent bits (" << exponentBits
               << ") exceeds the limit ([1, " << max_exponent_bits << "])";

    // Bias must be within limits.
    if (base2::getBitWidth(bias) > exponentBits) {
        const auto maxBias = (std::int64_t(1) << exponentBits) - 1;
        return emitError()
               << "bias (" << bias << ") exceeds limit (" << maxBias << ")";
    }

    // Total bit width must fit into a standard integer type.
    const auto bitWidth = precision + exponentBits;
    if (bitWidth > IntegerType::kMaxWidth)
        return emitError()
               << "total bits (" << bitWidth << ") exceed the limit ("
               << IntegerType::kMaxWidth << ")";

    return success();
}

void IEEE754Type::print(AsmPrinter &printer) const
{
    // `<`
    printer << "<";

    // $precision `,` $exponentBits
    printer << getPrecision() << ", " << getExponentBits();
    // (`,` $bias)?
    if (getBias() != base2::getBias(getExponentBits()))
        printer << ", " << getBias();

    // `>`
    printer << ">";
}

Type IEEE754Type::parse(AsmParser &parser)
{
    // `<`
    if (parser.parseLess()) return Type{};

    // $precision `,` $exponentBits
    bit::bit_width_t precision, exponentBits;
    if (parser.parseInteger(precision) || parser.parseComma()
        || parser.parseInteger(exponentBits))
        return Type{};

    // We must check this now to calculate the default bias, even though the
    // verifier also checks this.
    if (exponentBits < 1 || exponentBits > max_exponent_bits) {
        parser.emitError(parser.getNameLoc())
            << "exponent bits (" << exponentBits << ") exceeds the limit ([1, "
            << max_exponent_bits << "])";
        return Type{};
    }

    // (`,` $bias)?
    auto bias = base2::getBias(exponentBits);
    if (succeeded(parser.parseOptionalComma()))
        if (parser.parseInteger(bias)) return Type{};

    // `>`
    if (parser.parseGreater()) return Type{};

    return parser.getChecked<IEEE754Type>(
        parser.getContext(),
        precision,
        exponentBits,
        bias);
}

//===----------------------------------------------------------------------===//
// Base2Dialect
//===----------------------------------------------------------------------===//

void Base2Dialect::registerTypes()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "base2-mlir/Dialect/Base2/IR/Types.cpp.inc"
        >();
}

void Base2Dialect::printType(Type type, DialectAsmPrinter &printer) const
{
    // Print keyword abbreviation.
    if (const auto fixed = type.dyn_cast<FixedPointSemantics>()) {
        fixed.print(printer.getStream());
        return;
    }
    if (const auto ieee754 = type.dyn_cast<IEEE754Semantics>()) {
        ieee754.print(printer.getStream());
        return;
    }

    // Print using mnemonic.
    if (failed(generatedTypePrinter(type, printer)))
        llvm_unreachable("unexpected 'base2' type kind");
}

Type Base2Dialect::parseType(DialectAsmParser &parser) const
{
    StringRef typeTag;

    // Parse using mnemonic.
    Type genType;
    auto parseResult = generatedTypeParser(parser, &typeTag, genType);
    if (parseResult.has_value()) return genType;

    // Parse abbreviation.
    if (const auto fixed =
            FixedPointSemantics::parse(parser.getContext(), typeTag))
        return fixed;
    if (const auto ieee754 =
            IEEE754Semantics::parse(parser.getContext(), typeTag))
        return ieee754;

    parser.emitError(parser.getNameLoc(), "unknown base2 type: ") << typeTag;
    return {};
}
