/// Implements the Base2 dialect InterpretableType interface.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "base2-mlir/Dialect/Base2/Interfaces/InterpretableType.h"

#include "base2-mlir/Dialect/Base2/Analysis/IntInterpreter.h"
#include "base2-mlir/Dialect/Base2/IR/Base2.h"

#include "llvm/ADT/APSInt.h"

using namespace mlir;
using namespace mlir::base2;

bool interpretable_type_interface_defaults::canValueCast(
    Type self,
    Type from,
    Type to)
{
    const auto impl = self.cast<InterpretableType>();
    const auto fromImpl = from.cast<InterpretableType>();
    const auto toImpl = to.cast<InterpretableType>();

    return impl
        .valueCast(
            fromImpl,
            BitSequence::zeros(fromImpl.getBitWidth()),
            toImpl,
            RoundingMode::None)
        .has_value();
}

std::optional<BitSequence> interpretable_type_interface_defaults::min(
    Type self,
    const BitSequence &lhs,
    const BitSequence &rhs)
{
    const auto impl = self.cast<InterpretableType>();
    if (const auto cmp = impl.cmp(lhs, rhs)) {
        if (std::is_lteq(*cmp)) return lhs;
        if (std::is_gt(*cmp)) return rhs;
    }

    return std::nullopt;
}

std::optional<BitSequence> interpretable_type_interface_defaults::max(
    Type self,
    const BitSequence &lhs,
    const BitSequence &rhs)
{
    const auto impl = self.cast<InterpretableType>();
    if (const auto cmp = impl.cmp(lhs, rhs)) {
        if (std::is_gteq(*cmp)) return lhs;
        if (std::is_lt(*cmp)) return rhs;
    }

    return std::nullopt;
}

std::optional<std::partial_ordering> interpretable_type_interface_defaults::cmp(
    Type,
    const BitSequence &lhs,
    const BitSequence &rhs)
{
    if (lhs == rhs) return std::partial_ordering::equivalent;
    return std::nullopt;
}

//===- Generated implementation -------------------------------------------===//

#include "base2-mlir/Dialect/Base2/Interfaces/InterpretableType.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// External models
//===----------------------------------------------------------------------===//

namespace {

template<class Float>
struct FloatModel : InterpretableType::ExternalModel<FloatModel<Float>, Float> {
    static APFloat interpret(FloatType as, const BitSequence &value)
    {
        return APFloat(as.getFloatSemantics(), value.asUInt());
    }
    static APFloat interpret(Type as, const BitSequence &value)
    {
        return interpret(as.cast<FloatType>(), value);
    }

    static bit_result valueCastTo(
        APFloat value,
        InterpretableType to,
        llvm::RoundingMode roundingMode)
    {
        if (auto toFloat = to.dyn_cast<FloatType>()) {
            bool lossy;
            value.convert(toFloat.getFloatSemantics(), roundingMode, &lossy);
            return value;
        }

        if (const auto toInt = to.dyn_cast<IntegerType>()) {
            if (toInt.isSignless()) return std::nullopt;

            bool exact;
            llvm::APSInt result(toInt.getWidth(), toInt.isUnsigned());
            value.convertToInteger(result, roundingMode, &exact);
            return result;
        }

        return std::nullopt;
    }
    static bit_result valueCastFrom(
        InterpretableType from,
        const BitSequence &value,
        FloatType to,
        llvm::RoundingMode roundingMode)
    {
        if (const auto fromInt = from.dyn_cast<IntegerType>()) {
            if (fromInt.isSignless()) return std::nullopt;
            APFloat result(to.getFloatSemantics());
            result.convertFromAPInt(
                value.asUInt(),
                fromInt.isSigned(),
                roundingMode);
            return result;
        }

        return std::nullopt;
    }

    static bit_result valueCast(
        Type,
        InterpretableType from,
        const BitSequence &value,
        InterpretableType to,
        RoundingMode roundingMode)
    {
        const auto llvmRM = getLLVMRoundingMode(roundingMode);
        if (llvmRM == llvm::RoundingMode::Invalid) return std::nullopt;

        const auto fromImpl = from.dyn_cast<FloatType>();
        const auto toImpl = to.dyn_cast<FloatType>();

        if (fromImpl)
            return valueCastTo(interpret(fromImpl, value), to, llvmRM);
        if (toImpl) return valueCastFrom(from, value, toImpl, llvmRM);

        return std::nullopt;
    }

    static cmp_result
    cmp(Type self, const BitSequence &lhs, const BitSequence &rhs)
    {
        const auto lhsVal = interpret(self, lhs);
        const auto rhsVal = interpret(self, rhs);

        switch (lhsVal.compare(rhsVal)) {
        case APFloat::cmpEqual: return std::partial_ordering::equivalent;
        case APFloat::cmpGreaterThan: return std::partial_ordering::greater;
        case APFloat::cmpLessThan: return std::partial_ordering::less;
        case APFloat::cmpUnordered: return std::partial_ordering::unordered;
        };
    }

    static bit_result
    min(Type self, const BitSequence &lhs, const BitSequence &rhs)
    {
        const auto lhsVal = interpret(self, lhs);
        const auto rhsVal = interpret(self, rhs);

        if (lhsVal.isNaN()) return lhs;
        if (rhsVal.isNaN()) return rhs;

        return lhsVal <= rhsVal ? lhs : rhs;
    }

    static bit_result
    max(Type self, const BitSequence &lhs, const BitSequence &rhs)
    {
        const auto lhsVal = interpret(self, lhs);
        const auto rhsVal = interpret(self, rhs);

        if (lhsVal.isNaN()) return lhs;
        if (rhsVal.isNaN()) return rhs;

        return lhsVal >= rhsVal ? lhs : rhs;
    }

#define CLOSED_OP(op, call)                                                    \
    static bit_result op(                                                      \
        Type self,                                                             \
        const BitSequence &lhs,                                                \
        const BitSequence &rhs,                                                \
        RoundingMode roundingMode)                                             \
    {                                                                          \
        const auto llvmRM = getLLVMRoundingMode(roundingMode);                 \
        if (llvmRM == llvm::RoundingMode::Invalid) return std::nullopt;        \
                                                                               \
        auto lhsVal = interpret(self, lhs);                                    \
        const auto rhsVal = interpret(self, rhs);                              \
                                                                               \
        lhsVal.call(rhsVal, llvmRM);                                           \
        return lhsVal;                                                         \
    }

    CLOSED_OP(add, add)
    CLOSED_OP(sub, subtract)
    CLOSED_OP(mul, multiply)
    CLOSED_OP(div, divide)

#undef CLOSED_OP

    static bit_result
    mod(Type self, const BitSequence &lhs, const BitSequence &rhs)
    {
        auto lhsVal = interpret(self, lhs);
        const auto rhsVal = interpret(self, rhs);

        lhsVal.mod(rhsVal);
        return lhsVal;
    }

    static ValueFacts getFacts(Type self, const BitSequence &value)
    {
        const auto fltVal = interpret(self, value);
        auto facts = ValueFacts::None;

        // Handle sign.
        facts |=
            fltVal.isNegative() ? ValueFacts::Negative : ValueFacts::Positive;

        // Handle non-finite numbers.
        if (fltVal.isNaN()) return facts |= ValueFacts::NaN;
        if (fltVal.isInfinity()) {
            facts |= ValueFacts::Inf;
            facts |= fltVal.isNegative() ? ValueFacts::Min : ValueFacts::Max;
            return facts;
        }

        // Handle zero.
        if (fltVal.isZero()) return facts |= ValueFacts::Zero;

        // Handle one.
        const auto mantBits =
            APFloat::semanticsPrecision(fltVal.getSemantics()) - 1;
        const auto expBits =
            APFloat::semanticsSizeInBits(fltVal.getSemantics()) - mantBits - 1;
        if (value.asUInt().lshr(mantBits).trunc(expBits).isMaxSignedValue()) {
            if (value.asUInt().trunc(mantBits).isZero())
                return facts |= ValueFacts::One;
        }

        return facts;
    }
};

struct IntModel : InterpretableType::ExternalModel<IntModel, IntegerType> {
    static bit_result valueCast(
        Type,
        InterpretableType from,
        const BitSequence &value,
        InterpretableType to,
        RoundingMode roundingMode)
    {
        const auto fromImpl = from.dyn_cast<IntegerType>();
        const auto toImpl = to.dyn_cast<IntegerType>();
        if (!fromImpl || !toImpl) return std::nullopt;
        if (fromImpl.isSignless() || toImpl.isSignless()) return std::nullopt;

        return IntInterpreter::cast(
                   fromImpl.isSigned(),
                   value.asUInt(),
                   toImpl.isSigned(),
                   toImpl.getWidth(),
                   roundingMode)
            .getValue();
    }

    static cmp_result
    cmp(Type self, const BitSequence &lhs, const BitSequence &rhs)
    {
        const auto selfImpl = self.cast<IntegerType>();
        if (selfImpl.isSignless()) return std::nullopt;

        return IntInterpreter::cmp(
            selfImpl.isSigned(),
            lhs.asUInt(),
            rhs.asUInt());
    }

#define CLOSED_OP(op)                                                          \
    static bit_result op(                                                      \
        Type self,                                                             \
        const BitSequence &lhs,                                                \
        const BitSequence &rhs,                                                \
        RoundingMode roundingMode)                                             \
    {                                                                          \
        const auto selfImpl = self.cast<IntegerType>();                        \
        if (selfImpl.isSignless()) return std::nullopt;                        \
                                                                               \
        return IntInterpreter::op(                                             \
                   selfImpl.isSigned(),                                        \
                   lhs.asUInt(),                                               \
                   rhs.asUInt(),                                               \
                   roundingMode)                                               \
            .getValue();                                                       \
    }

    CLOSED_OP(add)
    CLOSED_OP(sub)
    CLOSED_OP(mul)
    CLOSED_OP(div)

#undef CLOSED_OP

    static bit_result
    mod(Type self, const BitSequence &lhs, const BitSequence &rhs)
    {
        const auto selfImpl = self.cast<IntegerType>();
        if (selfImpl.isSignless()) return std::nullopt;

        return selfImpl.isSigned() ? lhs.asUInt().srem(rhs.asUInt())
                                   : lhs.asUInt().urem(rhs.asUInt());
    }

    static ValueFacts getFacts(Type self, const BitSequence &value)
    {
        const auto selfImpl = self.cast<IntegerType>();

        return IntInterpreter::getFacts(
            getSignedness(selfImpl.getSignedness()),
            value.asUInt());
    }
};

} // namespace

void base2::registerInterpretableTypeModels(MLIRContext &ctx)
{
    Float8E5M2Type::attachInterface<FloatModel<Float8E5M2Type>>(ctx);
    BFloat16Type::attachInterface<FloatModel<BFloat16Type>>(ctx);
    Float16Type::attachInterface<FloatModel<Float16Type>>(ctx);
    Float32Type::attachInterface<FloatModel<Float32Type>>(ctx);
    Float64Type::attachInterface<FloatModel<Float64Type>>(ctx);
    Float80Type::attachInterface<FloatModel<Float80Type>>(ctx);
    Float128Type::attachInterface<FloatModel<Float128Type>>(ctx);

    IntegerType::attachInterface<IntModel>(ctx);
}
