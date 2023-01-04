/// Implements the constant folding dispatcher.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "base2-mlir/Dialect/Base2/Analysis/BitInterpreter.h"

#include "llvm/Support/Debug.h"

#include <numeric>

#define DEBUG_TYPE "base2-bitinterpreter"

using namespace mlir;
using namespace mlir::base2;

/// Writes @p value to @p out .
static llvm::raw_ostream &write(llvm::raw_ostream &out, bool value)
{
    return out << (value ? "true" : "false");
}
/// Writes @p opt to @p out .
static llvm::raw_ostream &
write(llvm::raw_ostream &out, const std::optional<BitSequence> &opt)
{
    if (!opt) return out << "{}";
    return out << *opt;
}
/// Writes @p opt to @p ordering .
static llvm::raw_ostream &write(
    llvm::raw_ostream &out,
    const std::optional<std::partial_ordering> ordering)
{
    if (!ordering) return out << "{}";
    if (std::is_eq(*ordering)) return out << "eq";
    if (std::is_lt(*ordering)) return out << "lt";
    if (std::is_gt(*ordering)) return out << "gt";
    return out << "unordered";
}
/// Writes @p facts to @p out .
static llvm::raw_ostream &write(llvm::raw_ostream &out, ValueFacts facts)
{
    return out << facts;
}

static void
interleaveComma(llvm::raw_ostream &out, auto &&head, auto &&... tail)
{
    out << head;
    if constexpr (sizeof...(tail) > 0) {
        out << ", ";
        interleaveComma(out, std::forward<decltype(tail)>(tail)...);
    }
}

static auto
trace(StringRef op, auto fn, InterpretableType type, auto &&... args)
{
    LLVM_DEBUG(
        llvm::dbgs() << op << "("; llvm::dbgs() << type << ": ";
        interleaveComma(llvm::dbgs(), std::forward<decltype(args)>(args)...);
        llvm::dbgs() << ") = ");

    const auto result = fn(type, std::forward<decltype(args)>(args)...);

    LLVM_DEBUG(write(llvm::dbgs(), result) << "\n");

    return result;
}

#define TRACE(op, ...)                                                         \
    trace(                                                                     \
        #op,                                                                   \
        [](auto type, auto &&... args) {                                       \
            return type.op(std::forward<decltype(args)>(args)...);             \
        },                                                                     \
        __VA_ARGS__)

//===----------------------------------------------------------------------===//
// BitInterpreter
//===----------------------------------------------------------------------===//

bool BitInterpreter::canValueCast(InterpretableType from, InterpretableType to)
{
    assert(from && to);

    // Try symmetrically.
    if (TRACE(canValueCast, from, from, to)) return true;
    if (TRACE(canValueCast, to, from, to)) return true;

    return false;
}

bit_result BitInterpreter::valueCast(
    InterpretableType from,
    const BitSequence &value,
    InterpretableType to,
    RoundingMode roundingMode)
{
    assert(from && to);
    assert(value.size() == from.getBitWidth());

    // Try symmetrically.
    if (const auto result =
            TRACE(valueCast, from, from, value, to, roundingMode))
        return result;
    if (const auto result = TRACE(valueCast, to, from, value, to, roundingMode))
        return result;

    return std::nullopt;
}

cmp_result BitInterpreter::cmp(
    InterpretableType type,
    const BitSequence &lhs,
    const BitSequence &rhs)
{
    assert(type);
    assert(lhs.size() == type.getBitWidth());
    assert(rhs.size() == type.getBitWidth());

    return TRACE(cmp, type, lhs, rhs);
}

bit_result BitInterpreter::min(
    InterpretableType type,
    const BitSequence &lhs,
    const BitSequence &rhs)
{
    assert(type);
    assert(lhs.size() == type.getBitWidth());
    assert(rhs.size() == type.getBitWidth());

    return TRACE(min, type, lhs, rhs);
}

bit_result BitInterpreter::max(
    InterpretableType type,
    const BitSequence &lhs,
    const BitSequence &rhs)
{
    assert(type);
    assert(lhs.size() == type.getBitWidth());
    assert(rhs.size() == type.getBitWidth());

    return TRACE(max, type, lhs, rhs);
}

bit_result BitInterpreter::add(
    InterpretableType type,
    const BitSequence &lhs,
    const BitSequence &rhs,
    RoundingMode roundingMode)
{
    assert(type);
    assert(lhs.size() == type.getBitWidth());
    assert(rhs.size() == type.getBitWidth());

    return TRACE(add, type, lhs, rhs, roundingMode);
}

bit_result BitInterpreter::sub(
    InterpretableType type,
    const BitSequence &lhs,
    const BitSequence &rhs,
    RoundingMode roundingMode)
{
    assert(type);
    assert(lhs.size() == type.getBitWidth());
    assert(rhs.size() == type.getBitWidth());

    return TRACE(sub, type, lhs, rhs, roundingMode);
}

bit_result BitInterpreter::mul(
    InterpretableType type,
    const BitSequence &lhs,
    const BitSequence &rhs,
    RoundingMode roundingMode)
{
    assert(type);
    assert(lhs.size() == type.getBitWidth());
    assert(rhs.size() == type.getBitWidth());

    return TRACE(mul, type, lhs, rhs, roundingMode);
}

bit_result BitInterpreter::div(
    InterpretableType type,
    const BitSequence &lhs,
    const BitSequence &rhs,
    RoundingMode roundingMode)
{
    assert(type);
    assert(lhs.size() == type.getBitWidth());
    assert(rhs.size() == type.getBitWidth());

    return TRACE(div, type, lhs, rhs, roundingMode);
}

bit_result BitInterpreter::mod(
    InterpretableType type,
    const BitSequence &lhs,
    const BitSequence &rhs)
{
    assert(type);
    assert(lhs.size() == type.getBitWidth());
    assert(rhs.size() == type.getBitWidth());

    return TRACE(mod, type, lhs, rhs);
}

ValueFacts
BitInterpreter::getFacts(InterpretableType type, const BitSequence &value)
{
    assert(type);
    assert(value.size() == type.getBitWidth());

    return TRACE(getFacts, type, value);
}

ValueFacts BitInterpreter::getFacts(BitSequenceLikeAttr attr)
{
    if (!attr) return ValueFacts::None;
    const auto impl = attr.getElementType().dyn_cast<InterpretableType>();
    if (!impl) return ValueFacts::None;

    if (const auto single = attr.dyn_cast<BitSequenceAttr>())
        return getFacts(impl, single.getValue());

    const auto dense = attr.cast<DenseBitSequencesAttr>();
    if (dense.isSplat()) return getFacts(impl, dense.getSplatValue());
    if (dense.empty()) return ValueFacts::None;
    return std::accumulate(
        dense.value_begin(),
        dense.value_end(),
        ValueFacts::All,
        [&](auto lhs, const auto &rhs) { return lhs & getFacts(impl, rhs); });
}
