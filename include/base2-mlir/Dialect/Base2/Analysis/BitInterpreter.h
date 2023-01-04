/// Declares the constant folding dispatcher.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "base2-mlir/Dialect/Base2/Analysis/BitSequence.h"
#include "base2-mlir/Dialect/Base2/Enums.h"
#include "base2-mlir/Dialect/Base2/Interfaces/BitSequenceAttr.h"
#include "base2-mlir/Dialect/Base2/Interfaces/InterpretableType.h"

#include <compare>
#include <optional>

namespace mlir::base2 {

/// Dispatcher for constant folding.
///
/// Instead of directly manipulating bit sequences and attributes and handling
/// the underlying implementation InterpretableType, this helper provides all
/// necessary overloads and convenience functions.
///
/// Additionally, all calls to interface methods are traced to the debug output,
/// so that constant folding problems can be debugged.
class BitInterpreter {
    static auto tryInterpret(auto fn, Type type, auto &&... args)
    {
        using result_type = decltype(fn(
            std::declval<InterpretableType>(),
            std::forward<decltype(args)>(args)...));

        if (const auto impl = type.dyn_cast<InterpretableType>())
            return fn(impl, std::forward<decltype(args)>(args)...);

        return result_type{};
    }
    static BitSequenceLikeAttr
    tryInterpret(auto fn, Attribute lhs, Attribute rhs, auto &&... args)
    {
        const auto lhsBits = lhs.dyn_cast_or_null<BitSequenceLikeAttr>();
        const auto rhsBits = rhs.dyn_cast_or_null<BitSequenceLikeAttr>();
        if (lhsBits && rhsBits)
            return fn(lhsBits, rhsBits, std::forward<decltype(args)>(args)...);

        return BitSequenceLikeAttr{};
    }

    [[nodiscard]] static BitSequenceLikeAttr tryInterpretBinary(
        auto fn,
        BitSequenceLikeAttr lhs,
        BitSequenceLikeAttr rhs,
        auto... args)
    {
        if (!lhs || !rhs) return BitSequenceLikeAttr{};
        const auto impl = lhs.getElementType().dyn_cast<InterpretableType>();
        if (!impl) return BitSequenceLikeAttr{};
        if (rhs.getElementType() != impl) return BitSequenceLikeAttr{};

        return lhs.zip(
            [&](const auto &l, const auto &r) {
                return fn(impl, l, r, args...);
            },
            rhs);
    }

#define DELEGATE(op)                                                           \
    [](auto &&... args) { return op(std::forward<decltype(args)>(args)...); }

public:
    //===------------------------------------------------------------------===//
    // Casting
    //===------------------------------------------------------------------===//

    /// Determines whether value_cast is permissible.
    ///
    /// @pre    `from`
    /// @pre    `to`
    [[nodiscard]] static bool
    canValueCast(InterpretableType from, InterpretableType to);
    [[nodiscard]] static bool canValueCast(Type from, Type to)
    {
        const auto fromImpl = from.dyn_cast<InterpretableLikeType>();
        const auto toImpl = to.dyn_cast<InterpretableLikeType>();
        return fromImpl && toImpl
               && canValueCast(
                   fromImpl.getElementType(),
                   toImpl.getElementType());
    }

    /// Implements value_cast on constant bit sequences.
    ///
    /// @pre    `from`
    /// @pre    `to`
    /// @pre    `value.size() == from.getBitWidth()`
    [[nodiscard]] static bit_result valueCast(
        InterpretableType from,
        const BitSequence &value,
        InterpretableType to,
        RoundingMode roundingMode = RoundingMode::None);
    [[nodiscard]] static bit_result valueCast(
        Type from,
        const BitSequence &value,
        Type to,
        RoundingMode roundingMode = RoundingMode::None)
    {
        const auto fromImpl = from.dyn_cast<InterpretableType>();
        const auto toImpl = to.dyn_cast<InterpretableType>();
        if (fromImpl && toImpl)
            return valueCast(fromImpl, value, toImpl, roundingMode);

        return std::nullopt;
    }
    [[nodiscard]] static BitSequenceLikeAttr valueCast(
        BitSequenceLikeAttr from,
        InterpretableType to,
        RoundingMode roundingMode = RoundingMode::None)
    {
        if (!from) return BitSequenceLikeAttr{};
        const auto impl = from.getElementType().dyn_cast<InterpretableType>();
        if (!impl) return BitSequenceLikeAttr{};

        return from.map(
            [&](const auto &l) { return valueCast(impl, l, to, roundingMode); },
            to.cast<BitSequenceType>());
    }
    [[nodiscard]] static BitSequenceLikeAttr valueCast(
        Attribute from,
        Type to,
        RoundingMode roundingMode = RoundingMode::None)
    {
        const auto fromBits = from.dyn_cast_or_null<BitSequenceLikeAttr>();
        const auto toImpl = to.dyn_cast<InterpretableLikeType>();
        if (fromBits && toImpl)
            return valueCast(fromBits, toImpl.getElementType(), roundingMode);

        return BitSequenceLikeAttr{};
    }

    //===------------------------------------------------------------------===//
    // Common operations
    //===------------------------------------------------------------------===//

    /// Implements cmp on constant bit sequences.
    ///
    /// @pre    `type`
    /// @pre    `lhs.size() == type.getBitWidth()`
    /// @pre    `rhs.size() == type.getBitWidth()`
    [[nodiscard]] static cmp_result
    cmp(InterpretableType type, const BitSequence &lhs, const BitSequence &rhs);
    [[nodiscard]] static cmp_result
    cmp(Type type, const BitSequence &lhs, const BitSequence &rhs)
    {
        return tryInterpret(DELEGATE(cmp), type, lhs, rhs);
    }
    [[nodiscard]] static BitSequenceLikeAttr
    cmp(PartialOrderingPredicate pred,
        BitSequenceLikeAttr lhs,
        BitSequenceLikeAttr rhs)
    {
        if (!lhs || !rhs) return BitSequenceLikeAttr{};
        const auto impl = lhs.getElementType().dyn_cast<InterpretableType>();
        if (!impl) return BitSequenceLikeAttr{};
        if (rhs.getElementType() != impl) return BitSequenceLikeAttr{};

        const auto i1Ty = IntegerType::get(lhs.getContext(), 1);
        return lhs.zip(
            [pred, impl](const auto &lhs, const auto &rhs) -> bit_result {
                if (const auto ordering = cmp(impl, lhs, rhs))
                    return BitSequence(matches(*ordering, pred));

                return std::nullopt;
            },
            rhs,
            i1Ty);
    }
    [[nodiscard]] static BitSequenceLikeAttr
    cmp(PartialOrderingPredicate pred, Attribute lhs, Attribute rhs)
    {
        const auto lhsBits = lhs.dyn_cast_or_null<BitSequenceLikeAttr>();
        const auto rhsBits = rhs.dyn_cast_or_null<BitSequenceLikeAttr>();
        if (lhsBits && rhsBits) return cmp(pred, lhsBits, rhsBits);

        return BitSequenceLikeAttr{};
    }

#define COMMON_OP(op)                                                          \
    [[nodiscard]] static bit_result op(                                        \
        InterpretableType type,                                                \
        const BitSequence &lhs,                                                \
        const BitSequence &rhs);                                               \
    [[nodiscard]] static bit_result op(                                        \
        Type type,                                                             \
        const BitSequence &lhs,                                                \
        const BitSequence &rhs)                                                \
    {                                                                          \
        return tryInterpret(DELEGATE(op), type, lhs, rhs);                     \
    }                                                                          \
    [[nodiscard]] static BitSequenceLikeAttr op(                               \
        BitSequenceLikeAttr lhs,                                               \
        BitSequenceLikeAttr rhs)                                               \
    {                                                                          \
        return tryInterpretBinary(DELEGATE(op), lhs, rhs);                     \
    }                                                                          \
    [[nodiscard]] static BitSequenceLikeAttr op(Attribute lhs, Attribute rhs)  \
    {                                                                          \
        return tryInterpret(DELEGATE(op), lhs, rhs);                           \
    }

    /// Implements min on constant bit sequences.
    ///
    /// @pre    `type`
    /// @pre    `lhs.size() == type.getBitWidth()`
    /// @pre    `rhs.size() == type.getBitWidth()`
    COMMON_OP(min)

    /// Implements max on constant bit sequences.
    ///
    /// @pre    `type`
    /// @pre    `lhs.size() == type.getBitWidth()`
    /// @pre    `rhs.size() == type.getBitWidth()`
    COMMON_OP(max)

    //===------------------------------------------------------------------===//
    // Closed arithmetic operations
    //===------------------------------------------------------------------===//

#define CLOSED_OP(op)                                                          \
    [[nodiscard]] static bit_result op(                                        \
        InterpretableType type,                                                \
        const BitSequence &lhs,                                                \
        const BitSequence &rhs,                                                \
        RoundingMode roundingMode = RoundingMode::None);                       \
    [[nodiscard]] static bit_result op(                                        \
        Type type,                                                             \
        const BitSequence &lhs,                                                \
        const BitSequence &rhs,                                                \
        RoundingMode roundingMode = RoundingMode::None)                        \
    {                                                                          \
        return tryInterpret(DELEGATE(op), type, lhs, rhs, roundingMode);       \
    }                                                                          \
    [[nodiscard]] static BitSequenceLikeAttr op(                               \
        BitSequenceLikeAttr lhs,                                               \
        BitSequenceLikeAttr rhs,                                               \
        RoundingMode roundingMode = RoundingMode::None)                        \
    {                                                                          \
        return tryInterpretBinary(DELEGATE(op), lhs, rhs, roundingMode);       \
    }                                                                          \
    [[nodiscard]] static BitSequenceLikeAttr op(                               \
        Attribute lhs,                                                         \
        Attribute rhs,                                                         \
        RoundingMode roundingMode = RoundingMode::None)                        \
    {                                                                          \
        return tryInterpret(DELEGATE(op), lhs, rhs, roundingMode);             \
    }

    /// Implements add on constant bit sequences.
    ///
    /// @pre    `type`
    /// @pre    `lhs.size() == type.getBitWidth()`
    /// @pre    `rhs.size() == type.getBitWidth()`
    CLOSED_OP(add)
    /// Implements sub on constant bit sequences.
    ///
    /// @pre    `type`
    /// @pre    `lhs.size() == type.getBitWidth()`
    /// @pre    `rhs.size() == type.getBitWidth()`
    CLOSED_OP(sub)
    /// Implements mul on constant bit sequences.
    ///
    /// @pre    `type`
    /// @pre    `lhs.size() == type.getBitWidth()`
    /// @pre    `rhs.size() == type.getBitWidth()`
    CLOSED_OP(mul)
    /// Implements div on constant bit sequences.
    ///
    /// @pre    `type`
    /// @pre    `lhs.size() == type.getBitWidth()`
    /// @pre    `rhs.size() == type.getBitWidth()`
    CLOSED_OP(div)
    /// Implements mod on constant bit sequences.
    ///
    /// @pre    `type`
    /// @pre    `lhs.size() == type.getBitWidth()`
    /// @pre    `rhs.size() == type.getBitWidth()`
    COMMON_OP(mod)

    //===------------------------------------------------------------------===//
    // Facts
    //===------------------------------------------------------------------===//

    /// Gets ValueFacts for a constant bit sequence.
    ///
    /// @pre    `type`
    /// @pre    `value.size() == from.getBitWidth()`
    [[nodiscard]] static ValueFacts
    getFacts(InterpretableType type, const BitSequence &value);
    [[nodiscard]] static ValueFacts
    getFacts(Type type, const BitSequence &value)
    {
        return tryInterpret(DELEGATE(getFacts), type, value);
    }
    [[nodiscard]] static ValueFacts getFacts(BitSequenceAttr attr)
    {
        if (!attr) return ValueFacts::None;
        const auto impl = attr.getType().dyn_cast<InterpretableType>();
        if (!impl) return ValueFacts::None;

        return getFacts(impl, attr.getValue());
    }
    [[nodiscard]] static ValueFacts getFacts(BitSequenceLikeAttr attr);

#undef CLOSED_OP
#undef COMMON_OP
#undef DELEGATE
};

} // namespace mlir::base2
