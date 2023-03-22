/// Implements the Base2 dialect IEEE754Semantics interface.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "base2-mlir/Dialect/Base2/Interfaces/IEEE754Semantics.h"

#include "base2-mlir/Dialect/Base2/IR/Base2.h"

#include <array>

using namespace mlir;
using namespace mlir::base2;

exponent_t ieee754_semantics_interface_defaults::getBias(Type self)
{
    const auto impl = self.cast<IEEE754Semantics>();
    return base2::getBias(impl.getExponentBits());
}

//===- Generated implementation -------------------------------------------===//

#include "base2-mlir/Dialect/Base2/Interfaces/IEEE754Semantics.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// IEEE754Semantics
//===----------------------------------------------------------------------===//

std::optional<APFloat::Semantics> IEEE754Semantics::toLLVMSemantics() const
{
    // Array of all semantics we could possibly match.
    static std::array semantics{
        &APFloat::IEEEhalf(),
        &APFloat::BFloat(),
        &APFloat::IEEEsingle(),
        &APFloat::IEEEdouble(),
        &APFloat::IEEEquad(),
        &APFloat::Float8E5M2(),
        &APFloat::x87DoubleExtended()};

    // Compare against every possible semantic (IEEE-754 only).
    const auto find = llvm::find_if(
        semantics,
        [&](const llvm::fltSemantics* sema) { return this->equals(sema); });

    // Return as an enum.
    if (find == semantics.end()) return std::nullopt;
    return APFloat::SemanticsToEnum(**find);
}

FloatType
IEEE754Semantics::getBuiltinType(MLIRContext* ctx, APFloat::Semantics llvmSema)
{
    assert(ctx);

    switch (llvmSema) {
    case APFloat::Semantics::S_IEEEhalf: return FloatType::getF16(ctx);
    case APFloat::Semantics::S_BFloat: return FloatType::getBF16(ctx);
    case APFloat::Semantics::S_IEEEsingle: return FloatType::getF32(ctx);
    case APFloat::Semantics::S_IEEEdouble: return FloatType::getF64(ctx);
    case APFloat::Semantics::S_IEEEquad: return FloatType::getF128(ctx);
    case APFloat::Semantics::S_x87DoubleExtended: return FloatType::getF80(ctx);
    case APFloat::Semantics::S_Float8E5M2: return FloatType::getFloat8E5M2(ctx);
    default: return FloatType{};
    }
}

IEEE754Semantics IEEE754Semantics::get(
    MLIRContext* ctx,
    bit::bit_width_t precision,
    bit::bit_width_t exponentBits,
    exponent_t bias)
{
    // NOTE: Returning a FloatType is of debatable use and unexpected.
    return IEEE754Type::get(ctx, precision, exponentBits, bias);
}

IEEE754Semantics IEEE754Semantics::parse(MLIRContext* ctx, StringRef str)
{
    assert(ctx);

    // `f`
    if (!str.consume_front("f")) return IEEE754Semantics{};

    // [0-9]+ `_` [0-9]+
    bit::bit_width_t precision, exponentBits;
    if (str.consumeInteger(10, precision)) return IEEE754Semantics{};
    if (!str.consume_front("_")) return IEEE754Semantics{};
    if (str.consumeInteger(10, exponentBits)) return IEEE754Semantics{};

    if (precision <= 2 || exponentBits < 1 || exponentBits > max_exponent_bits)
        return IEEE754Semantics{};
    if (str.empty())
        return get(ctx, precision, exponentBits, base2::getBias(exponentBits));

    // (`_` (`m`)? [0-9]+)?
    exponent_t biasSign = 1;
    std::uint32_t absBias;
    if (!str.consume_front("_")) return IEEE754Semantics{};
    if (str.consume_front("m")) biasSign = -1;
    if (str.consumeInteger(10, absBias)) return IEEE754Semantics{};

    if (!str.empty()) return IEEE754Semantics{};
    const auto bias = biasSign * static_cast<exponent_t>(absBias);
    if (base2::getBitWidth(bias) > exponentBits) return IEEE754Semantics{};

    return get(ctx, precision, exponentBits, bias);
}

void IEEE754Semantics::print(llvm::raw_ostream &out) const
{
    // `f`
    out << "f";

    // [0-9]+ `_` [0-9]+
    out << getPrecision() << "_" << getExponentBits();

    // (`_` (`m`)? [0-9]+)?
    if (getBias() == base2::getBias(getExponentBits())) return;
    out << "_";
    if (getBias() < 0) out << "m" << static_cast<std::uint32_t>(-getBias());
    out << static_cast<std::uint32_t>(getBias());
}

bool IEEE754Semantics::equals(const llvm::fltSemantics* rhs) const
{
    if (rhs == &APFloat::Float8E4M3FN()) return false;

    return getPrecision() == APFloat::semanticsPrecision(*rhs)
           && getBitWidth() == APFloat::semanticsSizeInBits(*rhs)
           && getMaxExponent() == APFloat::semanticsMaxExponent(*rhs)
           && getMinExponent() == APFloat::semanticsMinExponent(*rhs);
}

bool IEEE754Semantics::isSupersetOf(const llvm::fltSemantics* rhs) const
{
    if (rhs == &APFloat::Float8E4M3FN()) return false;

    return getPrecision() >= APFloat::semanticsPrecision(*rhs)
           && getMaxExponent() >= APFloat::semanticsMaxExponent(*rhs)
           && getMinExponent() <= APFloat::semanticsMinExponent(*rhs);
}

//===----------------------------------------------------------------------===//
// External models
//===----------------------------------------------------------------------===//

namespace {

template<class Float>
struct FloatModel : IEEE754Semantics::ExternalModel<FloatModel<Float>, Float> {
    static bit::bit_width_t getPrecision(Type type)
    {
        const auto &llvmSema = type.cast<FloatType>().getFloatSemantics();
        return APFloat::semanticsPrecision(llvmSema);
    }
    static bit::bit_width_t getExponentBits(Type type)
    {
        const auto &llvmSema = type.cast<FloatType>().getFloatSemantics();
        return APFloat::semanticsSizeInBits(llvmSema)
               - APFloat::semanticsPrecision(llvmSema);
    }
    static exponent_t getBias(Type type)
    {
        const auto &llvmSema = type.cast<FloatType>().getFloatSemantics();
        return exponent_t(1) - APFloat::semanticsMinExponent(llvmSema);
    }
};

} // namespace

void base2::registerIEEE754SemanticsModels(MLIRContext &ctx)
{
    Float8E5M2Type::attachInterface<FloatModel<Float8E5M2Type>>(ctx);
    BFloat16Type::attachInterface<FloatModel<BFloat16Type>>(ctx);
    Float16Type::attachInterface<FloatModel<Float16Type>>(ctx);
    Float32Type::attachInterface<FloatModel<Float32Type>>(ctx);
    Float64Type::attachInterface<FloatModel<Float64Type>>(ctx);
    Float80Type::attachInterface<FloatModel<Float80Type>>(ctx);
    Float128Type::attachInterface<FloatModel<Float128Type>>(ctx);
}
