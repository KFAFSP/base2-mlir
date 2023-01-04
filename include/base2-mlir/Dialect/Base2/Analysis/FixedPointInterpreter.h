/// Declares the constant folding interpreter for fixed-point numbers.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "base2-mlir/Dialect/Base2/Analysis/IntInterpreter.h"
#include "base2-mlir/Dialect/Base2/Interfaces/FixedPointSemantics.h"

#include <tuple>
#include <utility>

namespace mlir::base2 {

/// Implements constant folding for fixed-point numbers.
class FixedPointInterpreter {
public:
    /// Optional BitSequence result type.
    using bit_result = std::optional<BitSequence>;

    /// Optional BitSequence with associated FixedPointSemantics.
    struct [[nodiscard]] auto_result
            : std::pair<FixedPointSemantics, BitSequence> {
        /*implicit*/ auto_result() : pair() {}
        /*implicit*/ auto_result(std::nullopt_t) : auto_result() {}
        /*implicit*/ auto_result(FixedPointSemantics sema, BitSequence value)
                : pair(sema, std::move(value))
        {}

        FixedPointSemantics getSemantics() const { return first; }
        BitSequence &&getValue() && { return std::move(second); }
        const BitSequence &getValue() const & { return second; }

        /*implicit*/ operator FixedPointSemantics() const
        {
            return getSemantics();
        }
        /*implicit*/ operator bool() const
        {
            return static_cast<bool>(getSemantics());
        }
    };

    /// Optional pair of BitSequences with associated FixedPointSemantics.
    struct [[nodiscard]] align_result
            : std::tuple<FixedPointSemantics, BitSequence, BitSequence> {
        /*implicit*/ align_result() : tuple() {}
        /*implicit*/ align_result(std::nullopt_t) : align_result() {}
        /*implicit*/ align_result(
            FixedPointSemantics sema,
            BitSequence lhs,
            BitSequence rhs)
                : tuple(sema, std::move(lhs), std::move(rhs))
        {}

        FixedPointSemantics getSemantics() const { return std::get<0>(*this); }
        BitSequence &&getLhs() && { return std::get<1>(std::move(*this)); }
        const BitSequence &getLhs() const & { return std::get<1>(*this); }
        BitSequence &&getRhs() && { return std::get<2>(std::move(*this)); }
        const BitSequence &getRhs() const & { return std::get<2>(*this); }

        /*implicit*/ operator FixedPointSemantics() const
        {
            return getSemantics();
        }
        /*implicit*/ operator bool() const
        {
            return static_cast<bool>(getSemantics());
        }
    };

    //===------------------------------------------------------------------===//
    // Casting
    //===------------------------------------------------------------------===//

    /// Rescales @p value to @p outFracBits .
    ///
    /// @pre    `sema`
    /// @pre    `value.getBitWidth() == sema.getBitWidth()`
    static auto_result rescale(
        FixedPointSemantics sema,
        const BitSequence &value,
        bit_width_t outFracBits,
        RoundingMode roundingMode = RoundingMode::None);

    /// Converts @p value with @p from semantics to @p to semantics.
    ///
    /// @pre    `from`
    /// @pre    `to`
    /// @pre    `value.getBitWidth() == from.getBitWidth()`
    [[nodiscard]] static bit_result valueCast(
        FixedPointSemantics from,
        const BitSequence &value,
        FixedPointSemantics to,
        RoundingMode roundingMode = RoundingMode::None);

    //===------------------------------------------------------------------===//
    // Common operations
    //===------------------------------------------------------------------===//

    /// Compares @p lhs to @p rhs under @p sema .
    ///
    /// @pre    `sema`
    [[nodiscard]] static cmp_result
    cmp(FixedPointSemantics sema,
        const BitSequence &lhs,
        const BitSequence &rhs)
    {
        assert(sema);
        if (sema.isSignless()) return std::nullopt;

        return IntInterpreter::cmp(sema.isSigned(), lhs.asUInt(), rhs.asUInt());
    }

    /// Determines ValueFacts applying to @p value under @p sema .
    ///
    /// @pre    `sema`
    [[nodiscard]] static ValueFacts
    getFacts(FixedPointSemantics sema, const BitSequence &value)
    {
        assert(sema);

        return IntInterpreter::getFacts(sema.getSignedness(), value.asUInt());
    }

    //===------------------------------------------------------------------===//
    // Closed arithmetic operations
    //===------------------------------------------------------------------===//

    /// Adds @p lhs and @p rhs under @p sema with @p roundingMode .
    ///
    /// @pre    `sema`
    /// @pre    `lhs.size() == sema.getBitWidth()`
    /// @pre    `rhs.size() == sema.getBitWidth()`
    [[nodiscard]] static bit_result
    add(FixedPointSemantics sema,
        const BitSequence &lhs,
        const BitSequence &rhs,
        RoundingMode roundingMode = RoundingMode::None)
    {
        assert(sema);
        assert(lhs.size() == sema.getBitWidth());
        assert(rhs.size() == sema.getBitWidth());
        if (sema.isSignless()) return std::nullopt;

        return IntInterpreter::add(
                   sema.isSigned(),
                   lhs.asUInt(),
                   rhs.asUInt(),
                   roundingMode)
            .getValue();
    }
    /// Subtracts @p rhs from @p lhs under @p sema with @p roundingMode .
    ///
    /// @pre    `sema`
    /// @pre    `lhs.size() == sema.getBitWidth()`
    /// @pre    `rhs.size() == sema.getBitWidth()`
    [[nodiscard]] static bit_result
    sub(FixedPointSemantics sema,
        const BitSequence &lhs,
        const BitSequence &rhs,
        RoundingMode roundingMode = RoundingMode::None)
    {
        assert(sema);
        assert(lhs.size() == sema.getBitWidth());
        assert(rhs.size() == sema.getBitWidth());
        if (sema.isSignless()) return std::nullopt;

        return IntInterpreter::sub(
                   sema.isSigned(),
                   lhs.asUInt(),
                   rhs.asUInt(),
                   roundingMode)
            .getValue();
    }
    /// Multiplies @p lhs by @p rhs under @p sema with @p roundingMode .
    ///
    /// @pre    `sema`
    /// @pre    `lhs.size() == sema.getBitWidth()`
    /// @pre    `rhs.size() == sema.getBitWidth()`
    [[nodiscard]] static bit_result
    mul(FixedPointSemantics sema,
        const BitSequence &lhs,
        const BitSequence &rhs,
        RoundingMode roundingMode = RoundingMode::None)
    {
        if (const auto exact = mul(sema, lhs, sema, rhs))
            return valueCast(exact, exact.getValue(), sema, roundingMode);

        return std::nullopt;
    }
    /// Divides @p lhs by @p rhs under @p sema with @p roundingMode .
    ///
    /// @pre    `sema`
    /// @pre    `lhs.size() == sema.getBitWidth()`
    /// @pre    `rhs.size() == sema.getBitWidth()`
    [[nodiscard]] static bit_result
    div(FixedPointSemantics sema,
        const BitSequence &lhs,
        const BitSequence &rhs,
        RoundingMode roundingMode = RoundingMode::None)
    {
        if (const auto exact = div(sema, lhs, sema, rhs))
            return valueCast(exact, exact.getValue(), sema, roundingMode);

        return std::nullopt;
    }
    /// Computes the remainder of dividing @p lhs by @p rhs under @p sema with
    /// @p roundingMode .
    ///
    /// @pre    `sema`
    /// @pre    `lhs.size() == sema.getBitWidth()`
    /// @pre    `rhs.size() == sema.getBitWidth()`
    [[nodiscard]] static bit_result
    mod(FixedPointSemantics sema,
        const BitSequence &lhs,
        const BitSequence &rhs)
    {
        assert(sema);
        assert(lhs.size() == sema.getBitWidth());
        assert(rhs.size() == sema.getBitWidth());
        if (sema.isSignless()) return std::nullopt;

        return sema.isSigned() ? lhs.asUInt().srem(rhs.asUInt())
                               : lhs.asUInt().urem(rhs.asUInt());
    }

    //===------------------------------------------------------------------===//
    // Exact operations
    //===------------------------------------------------------------------===//

    /// Obtains the smallest supertype of @p lhs and @p rhs , if any.
    ///
    /// @pre    `lhs`
    /// @pre    `rhs`
    [[nodiscard]] static FixedPointSemantics
    align(FixedPointSemantics lhs, FixedPointSemantics rhs);
    /// Aligns @p lhs and @p rhs to a common supertype.
    ///
    /// @pre    `lhsSema`
    /// @pre    `lhs.size() == lhsSema.getBitWidth()`
    /// @pre    `rhsSema`
    /// @pre    `rhs.size() == lhsSema.getBitWidth()`
    static align_result align(
        FixedPointSemantics lhsSema,
        const BitSequence &lhs,
        FixedPointSemantics rhsSema,
        const BitSequence &rhs);

    /// Obtains the result semantics of adding @p lhs to @p rhs , if any.
    ///
    /// @pre    `lhs`
    /// @pre    `rhs`
    [[nodiscard]] static FixedPointSemantics
    add(FixedPointSemantics lhs, FixedPointSemantics rhs);
    /// Performs exact addition of @p lhs and @p rhs .
    ///
    /// @pre    `lhsSema`
    /// @pre    `lhs.size() == lhsSema.getBitWidth()`
    /// @pre    `rhsSema`
    /// @pre    `rhs.size() == lhsSema.getBitWidth()`
    static auto_result
    add(FixedPointSemantics lhsSema,
        const BitSequence &lhs,
        FixedPointSemantics rhsSema,
        const BitSequence &rhs);

    /// Obtains the result semantics of subtracting @p rhs from @p lhs , if any.
    ///
    /// @pre    `lhs`
    /// @pre    `rhs`
    [[nodiscard]] static FixedPointSemantics
    sub(FixedPointSemantics lhs, FixedPointSemantics rhs);
    /// Performs exact subtraction of @p rhs from @p lhs .
    ///
    /// @pre    `lhsSema`
    /// @pre    `lhs.size() == lhsSema.getBitWidth()`
    /// @pre    `rhsSema`
    /// @pre    `rhs.size() == lhsSema.getBitWidth()`
    static auto_result
    sub(FixedPointSemantics lhsSema,
        const BitSequence &lhs,
        FixedPointSemantics rhsSema,
        const BitSequence &rhs);

    /// Obtains the result semantics of multiplying @p lhs by @p rhs , if any.
    ///
    /// @pre    `lhs`
    /// @pre    `rhs`
    [[nodiscard]] static FixedPointSemantics
    mul(FixedPointSemantics lhs, FixedPointSemantics rhs);
    /// Performs exact multiplication of @p lhs by @p rhs .
    ///
    /// @pre    `lhsSema`
    /// @pre    `lhs.size() == lhsSema.getBitWidth()`
    /// @pre    `rhsSema`
    /// @pre    `rhs.size() == lhsSema.getBitWidth()`
    static auto_result
    mul(FixedPointSemantics lhsSema,
        const BitSequence &lhs,
        FixedPointSemantics rhsSema,
        const BitSequence &rhs);

    /// Obtains the result semantics of dividing @p lhs by @p rhs , if any.
    ///
    /// @pre    `lhs`
    /// @pre    `rhs`
    [[nodiscard]] static FixedPointSemantics
    div(FixedPointSemantics lhs, FixedPointSemantics rhs);
    /// Performs exact division of @p lhs by @p rhs .
    ///
    /// @pre    `lhsSema`
    /// @pre    `lhs.size() == lhsSema.getBitWidth()`
    /// @pre    `rhsSema`
    /// @pre    `rhs.size() == lhsSema.getBitWidth()`
    static auto_result
    div(FixedPointSemantics lhsSema,
        const BitSequence &lhs,
        FixedPointSemantics rhsSema,
        const BitSequence &rhs);
};

} // namespace mlir::base2

namespace std {

template<>
struct tuple_size<mlir::base2::FixedPointInterpreter::align_result>
        : integral_constant<size_t, 3> {};

template<>
struct tuple_element<0, mlir::base2::FixedPointInterpreter::align_result> {
    using type = mlir::base2::FixedPointSemantics;
};
template<>
struct tuple_element<1, mlir::base2::FixedPointInterpreter::align_result> {
    using type = mlir::base2::BitSequence;
};
template<>
struct tuple_element<2, mlir::base2::FixedPointInterpreter::align_result> {
    using type = mlir::base2::BitSequence;
};

} // namespace std
