/// Implements the Base2 dialect FixedPointSemantics interface.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "base2-mlir/Dialect/Base2/Interfaces/FixedPointSemantics.h"

#include "base2-mlir/Dialect/Base2/IR/Base2.h"

using namespace mlir;
using namespace mlir::base2;

//===- Generated implementation -------------------------------------------===//

#include "base2-mlir/Dialect/Base2/Interfaces/FixedPointSemantics.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// FixedPointSemantics
//===----------------------------------------------------------------------===//

FixedPointSemantics
FixedPointSemantics::get(IntegerType integerType, exponent_t exponent)
{
    // NOTE: This is of debatable use and unexpected.
    // if (fractionalBits == 0) return integerType.cast<FixedPointSemantics>();

    return FixedPointType::get(integerType, exponent);
}

FixedPointSemantics FixedPointSemantics::parse(MLIRContext* ctx, StringRef str)
{
    assert(ctx);

    // (`s` | `u`)? `i`
    Signedness signedness;
    if (str.consume_front("si"))
        signedness = Signedness::Signed;
    else if (str.consume_front("ui"))
        signedness = Signedness::Unsigned;
    else if (str.consume_front("i"))
        signedness = Signedness::Signless;
    else
        return FixedPointSemantics{};

    // [0-9]+
    bit_width_t bitWidth = 0;
    exponent_t exponent = 0;
    if (str.consumeInteger(10, bitWidth)) return FixedPointSemantics{};

    bool isNegative = true;
    // (`_` [0-9]+)?
    if (str.consume_front("_")) {
        // Parse the fractional bit width.
        if (str.consume_front("m")) isNegative = false;
        if (str.consumeInteger(10, exponent)) return FixedPointSemantics{};
    }

    if (!str.empty()) return FixedPointSemantics{};

    if (isNegative) exponent = 0 - exponent;

    return get(ctx, signedness, bitWidth, exponent);
}

void FixedPointSemantics::print(llvm::raw_ostream &out) const
{
    // (`s` | `u`)? `i`
    switch (getSignedness()) {
    case Signedness::Signed: out << "s"; break;
    case Signedness::Unsigned: out << "u"; break;
    default: break;
    }
    out << "i";

    // [0-9]+
    out << getBitWidth();

    // (`_` [0-9]+)?
    if (const auto exponent = getExponent()) {
        if (exponent > 0)
            out << "_m" << exponent;
        else if (exponent < 0)
            out << "_" << -exponent;
    }
}

bool FixedPointSemantics::isSupersetOf(FixedPointSemantics rhs) const
{
    assert(rhs);

    // For types with the same signedness, we can compare widths.
    if (getSignedness() == rhs.getSignedness())
        return getIntegerBits() >= rhs.getIntegerBits()
               && getFractionalBits() >= rhs.getFractionalBits();

    // For signless types, we can't say anything.
    if (isSignless() || rhs.isSignless()) return false;

    // A signed type encloses an unsigned type, iff
    if (isSigned()) {
        assert(rhs.isUnsigned());

        // There is one more integer bit, and at least as many fractional ones.
        return getIntegerBits() > rhs.getIntegerBits()
               && getFractionalBits() >= rhs.getFractionalBits();
    }

    // An unsigned type never encloses a signed type.
    assert(isUnsigned());
    return false;
}

//===----------------------------------------------------------------------===//
// External models
//===----------------------------------------------------------------------===//

namespace {

struct IntModel : FixedPointSemantics::ExternalModel<IntModel, IntegerType> {
    static IntegerType getIntegerType(Type type)
    {
        return type.cast<IntegerType>();
    }
};

} // namespace

void base2::registerFixedPointSemanticsModels(MLIRContext &ctx)
{
    IntegerType::attachInterface<IntModel>(ctx);
}
