/// Declares IR matchers for the Bit dialect.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "base2-mlir/Dialect/Base2/IR/Attributes.h"
#include "ub-mlir/Dialect/UB/IR/UB.h"
#include "ub-mlir/Dialect/UB/Utils/Matchers.h"

namespace mlir::bit::match {

/// Value that indicates a poisoned BitSequence.
static constexpr auto poison = std::nullopt;

/// Matches a constant (container of) BitSequence value(s).
using Const = BitSequenceLikeAttr;

/// Matches poison, or a constant (container of) BitSequence value(s).
struct ConstOrPoison : ub::ValueOrPoisonAttr<Const> {
    using ub::ValueOrPoisonAttr<Const>::ValueOrPoisonAttr;
    using ub::ValueOrPoisonAttr<Const>::get;

    /// Builds the canonical attribute for @p value and @p poisonMask .
    ///
    /// @pre    `value`
    [[nodiscard]] static ConstOrPoison
    get(Const value, const llvm::APInt &poisonMask)
    {
        assert(value);

        // Handle quick canonicalization cases.
        if (poisonMask.isAllOnes()) return ConstOrPoison::get(value.getType());
        if (poisonMask.isZero()) return value;

        // Make a partially poisoned value.
        return ConstOrPoison::get("bit", value, poisonMask);
    }
    /// Builds the canonical splat attribute for @p type and @p value .
    ///
    /// @pre    `type`
    /// @pre    `!value || value->size() >= type.getElementType().getBitWidth()`
    [[nodiscard]] static ConstOrPoison
    getSplat(BitSequenceLikeType type, const std::optional<BitSequence> &value)
    {
        assert(type);

        if (!value) return ConstOrPoison::get(type);
        return Const::getSplat(type, *value);
    }

    /*implicit*/ ConstOrPoison(Const attr) : ub::ValueOrPoisonAttr<Const>(attr)
    {}
    /*implicit*/ ConstOrPoison(ub::ValueOrPoisonAttr<Const> attr)
            : ub::ValueOrPoisonAttr<Const>(attr)
    {}

    using UnaryFn =
        std::optional<BitSequence>(const std::optional<BitSequence> &);
    using BinaryFn = std::optional<BitSequence>(
        const std::optional<BitSequence> &,
        const std::optional<BitSequence> &);
    using TernaryFn = std::optional<BitSequence>(
        const std::optional<BitSequence> &,
        const std::optional<BitSequence> &,
        const std::optional<BitSequence> &);

    [[nodiscard]] ConstOrPoison
    map(function_ref<UnaryFn> fn, BitSequenceType elementTy = {}) const;

    [[nodiscard]] ConstOrPoison
    zip(function_ref<BinaryFn> fn,
        ConstOrPoison rhs,
        BitSequenceType elementTy = {}) const;

    [[nodiscard]] ConstOrPoison
    zip(function_ref<TernaryFn> fn,
        ConstOrPoison arg1,
        ConstOrPoison arg2,
        BitSequenceType elementTy = {}) const;
};

/// Matches a constant Splat of BitSequence values.
struct Splat : Const {
    using Const::Const;

    [[nodiscard]] static bool classof(BitSequenceAttr) { return true; }
    [[nodiscard]] static bool classof(DenseBitSequencesAttr dense)
    {
        return dense.isSplat();
    }
    [[nodiscard]] static bool classof(Attribute attr)
    {
        if (llvm::isa<BitSequenceAttr>(attr)) return true;
        if (const auto dense = llvm::dyn_cast<DenseBitSequencesAttr>(attr))
            return classof(dense);
        return false;
    }

    /// Gets the canonical Splat attribute for @p type and @p value .
    ///
    /// @pre    `type`
    /// @pre    `value.size() >= type.getElementType().getBitWidth()`
    [[nodiscard]] static Splat
    get(BitSequenceLikeType type, const BitSequence &value)
    {
        return llvm::cast<Splat>(Const::getSplat(type, value));
    }

    /*implicit*/ Splat(BitSequenceAttr single) : Const(single) {}
    /*implicit*/ Splat(Const) = delete;

    /// Gets the splat BitSequence value.
    [[nodiscard]] BitSequence getValue() const
    {
        if (const auto single = llvm::dyn_cast<BitSequenceAttr>(*this))
            return single.getValue();
        return llvm::cast<DenseBitSequencesAttr>(*this).getSplatValue();
    }

    /// @copydoc getValue()
    /*implicit*/ operator BitSequence() const { return getValue(); }
};

/// Matches constant zeros.
struct Zeros : Splat {
    using Splat::Splat;

    [[nodiscard]] static bool classof(Splat splat)
    {
        return splat.getValue().isZeros();
    }
    [[nodiscard]] static bool classof(Attribute attr)
    {
        if (const auto splat = llvm::dyn_cast<Splat>(attr))
            return classof(splat);
        return false;
    }

    /*implicit*/ Zeros(Splat) = delete;
};

/// Matches constant ones.
struct Ones : Splat {
    using Splat::Splat;

    [[nodiscard]] static bool classof(Splat splat)
    {
        return splat.getValue().isOnes();
    }
    [[nodiscard]] static bool classof(Attribute attr)
    {
        if (const auto splat = llvm::dyn_cast<Splat>(attr))
            return classof(splat);
        return false;
    }

    /*implicit*/ Ones(Splat) = delete;
};

/// Matches poison, or a constant splat of BitSequence values.
struct SplatOrPoison : ConstOrPoison {
    using ConstOrPoison::ConstOrPoison;

    [[nodiscard]] static bool classof(ConstOrPoison attr)
    {
        return attr.isPoison()
               || (!attr.isPoisoned() && llvm::isa<Splat>(attr.getValueAttr()));
    }
    [[nodiscard]] static bool classof(Attribute attr)
    {
        if (const auto constOrPoison = llvm::dyn_cast<ConstOrPoison>(attr))
            return classof(constOrPoison);
        return false;
    }

    /// Gets the canonical SplatOrPoison attribute for @p type and @p value .
    ///
    /// @pre    `type`
    /// @pre    `!value || value->size() >= type.getElementType().getBitWidth()`
    [[nodiscard]] SplatOrPoison
    get(BitSequenceLikeType type, const std::optional<BitSequence> &value)
    {
        if (!value) return llvm::cast<SplatOrPoison>(ConstOrPoison::get(type));
        return llvm::cast<SplatOrPoison>(Const::getSplat(type, *value));
    }

    /*implicit*/ SplatOrPoison(BitSequenceAttr single) : ConstOrPoison(single)
    {}
    /*implicit*/ SplatOrPoison(ConstOrPoison) = delete;

    /// Gets the splat BitSequence value.
    [[nodiscard]] std::optional<BitSequence> getValue() const
    {
        if (this->isPoison()) return std::nullopt;
        return llvm::cast<Splat>(getSourceAttr()).getValue();
    }

    /// @copydoc getValue()
    /*implicit*/ operator std::optional<BitSequence>() const
    {
        return getValue();
    }
};

/// Matches a constant index value.
struct ConstIndex : IntegerAttr {
    using IntegerAttr::IntegerAttr;

    [[nodiscard]] static bool classof(IntegerAttr attr)
    {
        return attr.getType().isIndex();
    }
    [[nodiscard]] static bool classof(Attribute attr)
    {
        if (const auto intAttr = llvm::dyn_cast<IntegerAttr>(attr))
            return classof(intAttr);
        return false;
    }

    /*implicit*/ ConstIndex(IntegerAttr) = delete;

    [[nodiscard]] std::size_t getValue() const
    {
        return IntegerAttr::getValue().getZExtValue();
    }

    /*implicit*/ operator std::size_t() const { return getValue(); }
};

} // namespace mlir::bit::match
