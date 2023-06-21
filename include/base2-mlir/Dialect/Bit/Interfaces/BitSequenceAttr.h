/// Declares the Bit BitSequenceAttr interface.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "base2-mlir/Dialect/Bit/Analysis/BitSequence.h"
#include "base2-mlir/Dialect/Bit/Interfaces/BitSequenceType.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir::bit {

/// Reference to an unary BitSequence operation.
using UnaryBitSequenceFn = function_ref<BitSequence(const BitSequence &)>;

/// Reference to a binary BitSequence operation.
using BinaryBitSequenceFn =
    function_ref<BitSequence(const BitSequence &, const BitSequence &)>;

/// Reference to a ternary BitSequence operation.
using TernaryBitSequenceFn = function_ref<
    BitSequence(const BitSequence &, const BitSequence &, const BitSequence &)>;

} // namespace mlir::bit

//===- Generated includes -------------------------------------------------===//

#include "base2-mlir/Dialect/Bit/Interfaces/BitSequenceAttr.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::bit {

//===----------------------------------------------------------------------===//
// DenseBitSequencesAttr
//===----------------------------------------------------------------------===//

/// Concept for an ElementsAttr that stores densely packed BitSequences.
///
/// Satisfied by any ElementsAttr that:
///     - has a BitSequenceType element type
///     - supports iteration as
///         - BitSequence
///         - OR, if it has a FloatType element type, llvm::APFloat
///         - OR llvm::APInt
class DenseBitSequencesAttr : public ElementsAttr {
public:
    using ValueType = BitSequence;
    using iterator = mlir::detail::ElementsAttrIterator<ValueType>;
    using range = mlir::detail::ElementsAttrRange<iterator>;

    using ElementsAttr::ElementsAttr;

    /// @copydoc classof(Attribute)
    [[nodiscard]] static bool classof(ElementsAttr attr);
    /// Determines whether @p attr is a DenseBitSequencesAttr.
    [[nodiscard]] static bool classof(Attribute attr)
    {
        if (const auto elements = attr.dyn_cast<ElementsAttr>())
            return classof(elements);

        return false;
    }

    /// Builds the canonical data attribute for @p type and @p data .
    ///
    /// @pre    `type`
    /// @pre    `type.hasStaticShape()`
    /// @pre    `type.getElementType().isa<BitSequenceType>()`
    /// @pre    `values.size() == 1 || values.size() == type.getNumElements()`
    /// @pre    `value.size() == type.getBitWidth()`
    [[nodiscard]] static DenseIntOrFPElementsAttr
    getData(ShapedType type, ArrayRef<BitSequence> values);

    /// Builds the canonical DenseBitSequencesAttr for @p type using @p data .
    ///
    /// This function preserves the @p data attribute as the storage, performing
    /// at most bit casting when necessary.
    ///
    /// @pre    `type`
    /// @pre    `data`
    /// @pre    `type.getShape().equals(data.getType().getShape())`
    /// @pre    `type.getElementType().isa<BitSequenceType>()`
    /// @pre    `data.getElementBitWidth() == elementTy.getBitWidth()`
    [[nodiscard]] static DenseBitSequencesAttr
    get(ShapedType type, DenseIntOrFPElementsAttr data);
    /// Builds the canonical DenseBitSequencesAttr for @p type using @p values .
    ///
    /// When constructing a BitSequenceAttr, built-in attributes are
    /// prefered where possible. In all other scenarios, a DenseBitsAttr is
    /// returned.
    ///
    /// @pre    `type`
    /// @pre    `type.hasStaticShape()`
    /// @pre    `type.getElementType().isa<BitSequenceType>()`
    /// @pre    `values.size() == 1 || values.size() == type.getNumElements()`
    /// @pre    `value.size() == type.getBitWidth()`
    [[nodiscard]] static DenseBitSequencesAttr
    get(ShapedType type, ArrayRef<BitSequence> values)
    {
        return get(type, getData(type, values));
    }
    /// Builds the canonical DenseBitSequencesAttr using @p splatValue .
    ///
    /// @pre    `type`
    /// @pre    `type.hasStaticShape()`
    /// @pre    `type.getElementType().isa<BitSequenceType>()`
    /// @pre    `splatValue.size() == type.getBitWidth()`
    [[nodiscard]] static DenseBitSequencesAttr
    get(ShapedType type, const BitSequence &splatValue)
    {
        return get(type, getData(type, splatValue));
    }
    /// Builds the canonical DenseBitSequencesAttr for @p attr .
    ///
    /// @pre    `attr`
    [[nodiscard]] static DenseBitSequencesAttr get(DenseBitSequencesAttr attr)
    {
        assert(attr);

        if (attr.getElementType().isa<FloatType, IntegerType>()) {
            if (attr.isa<DenseIntOrFPElementsAttr>()) return attr;
        }

        return get(
            attr.getType().dyn_cast<ShapedType>(),
            llvm::to_vector(attr.getValues()));
    }

    /// Bit casts all elements.
    ///
    /// @pre    `elementTy`
    /// @pre    `getElementType().getBitWidth() == elementTy.getBitWidth()`
    [[nodiscard]] DenseBitSequencesAttr
    bitCastElements(BitSequenceType elementTy) const;

    /// Applies @p fn to the contained values.
    ///
    /// If @p elementTy is nullptr, getType() is used.
    ///
    /// @pre    bit width of @p elementTy and @p fn result matches
    [[nodiscard]] DenseBitSequencesAttr
    map(UnaryBitSequenceFn fn,
        BitSequenceType elementTy = {},
        bool allowSplat = true) const;
    /// Combines the values contained in this and @p rhs using @p fn .
    ///
    /// If @p elementTy is nullptr, getType() is used.
    ///
    /// @pre    `rhs`
    /// @pre    `rhs.getType().getShape().equals(getType().getShape())`
    /// @pre    bit width of @p elementTy and @p fn result matches
    [[nodiscard]] DenseBitSequencesAttr
    zip(BinaryBitSequenceFn fn,
        DenseBitSequencesAttr rhs,
        BitSequenceType elementTy = {},
        bool allowSplat = true) const;
    /// Combines the values contained in this, @p arg1 and @p arg2 using @p fn .
    ///
    /// If @p elementTy is nullptr, getType() is used.
    ///
    /// @pre    `arg1 && arg2`
    /// @pre    `arg1.getType().getShape().equals(getType().getShape())`
    /// @pre    `arg2.getType().getShape().equals(getType().getShape())`
    /// @pre    bit width of @p elementTy and @p fn result matches
    [[nodiscard]] DenseBitSequencesAttr
    zip(TernaryBitSequenceFn fn,
        DenseBitSequencesAttr arg1,
        DenseBitSequencesAttr arg2,
        BitSequenceType elementTy = {},
        bool allowSplat = true) const;

    //===------------------------------------------------------------------===//
    // ElementsAttr
    //===------------------------------------------------------------------===//

    /// Gets the BitSequenceType element type.
    [[nodiscard]] BitSequenceType getElementType() const
    {
        return ElementsAttr::getElementType().cast<BitSequenceType>();
    }

    /// Gets the BitSequence values.
    [[nodiscard]] range getValues() const
    {
        return range(
            getType().dyn_cast<ShapedType>(),
            value_begin(),
            value_end());
    }

    /// Gets the splat BitSequence value.
    BitSequence getSplatValue() const { return *value_begin(); }

    /// Gets the value begin iterator.
    [[nodiscard]] iterator value_begin() const;
    /// Gets the value end iterator.
    [[nodiscard]] iterator value_end() const { return iterator({}, size()); }
};

//===----------------------------------------------------------------------===//
// BitSequenceLikeAttr
//===----------------------------------------------------------------------===//

/// Concept for an attribute that declares compile-time constant bit sequences.
///
/// Satisfied by either a BitSequenceAttr or a DenseBitSequencesAttr.
class BitSequenceLikeAttr : public Attribute {
public:
    using Attribute::Attribute;

    /// @copydoc classof(Attribute)
    [[nodiscard]] static bool classof(BitSequenceAttr) { return true; }
    /// @copydoc classof(Attribute)
    [[nodiscard]] static bool classof(DenseBitSequencesAttr) { return true; }
    /// Determines whether @p attr is a BitSequenceLikeAttr.
    [[nodiscard]] static bool classof(Attribute attr)
    {
        if (attr.isa<BitSequenceAttr>()) return true;
        if (attr.isa<DenseBitSequencesAttr>()) return true;

        return false;
    }

    /// Builds the canonical BitSequenceLikeAttr for @p type and @p bits .
    ///
    /// @pre    `type`
    /// @pre    `bits.size() >= type.getBitWidth()`
    [[nodiscard]] static BitSequenceLikeAttr
    get(BitSequenceType type, const BitSequence &bits)
    {
        return BitSequenceAttr::get(type, bits);
    }
    /// Builds the canonical BitSequenceLikeAttr for @p type using @p values .
    ///
    /// @pre    `type`
    /// @pre    `type.hasStaticShape()`
    /// @pre    `type.getElementType().isa<BitSequenceType>()`
    /// @pre    `values.size() == 1 || values.size() == type.getNumElements()`
    /// @pre    `value.size() >= type.getBitWidth()`
    [[nodiscard]] static BitSequenceLikeAttr
    get(ShapedType type, ArrayRef<BitSequence> values)
    {
        return DenseBitSequencesAttr::get(type, values)
            .cast<BitSequenceLikeAttr>();
    }
    /// Builds the canonical BitSequenceLikeAttr for @p attr .
    ///
    /// @pre    `attr`
    [[nodiscard]] static BitSequenceLikeAttr get(BitSequenceLikeAttr attr)
    {
        if (const auto bits = attr.dyn_cast<BitSequenceAttr>())
            return BitSequenceAttr::get(bits);

        return DenseBitSequencesAttr::get(attr.cast<DenseBitSequencesAttr>());
    }

    /// Builds a splat BitSequenceLikeAttr with @p type and @p bits .
    ///
    /// @pre    `type`
    /// @pre    `bits.size() >= type.getElementType().getBitWidth()`
    [[nodiscard]] static BitSequenceLikeAttr
    getSplat(BitSequenceLikeType type, const BitSequence &bits)
    {
        if (auto elementTy = type.dyn_cast<BitSequenceType>())
            return get(elementTy, bits);

        return DenseBitSequencesAttr::get(type.cast<ShapedType>(), bits);
    }

    /// Initializes a BitSequenceLikeAttr from @p attr .
    ///
    /// @pre    `attr`
    /*implicit*/ BitSequenceLikeAttr(BitSequenceAttr attr)
            : Attribute(attr.cast<Attribute>().getImpl())
    {}
    /// Initializes a DenseBitSequencesAttr from @p attr .
    ///
    /// @pre    `attr`
    /*implicit*/ BitSequenceLikeAttr(DenseBitSequencesAttr attr)
            : Attribute(attr.cast<Attribute>().getImpl())
    {}

    /// Gets the type of this attribute.
    [[nodiscard]] BitSequenceLikeType getType() const
    {
        if (const auto bits = dyn_cast<BitSequenceAttr>())
            return bits.getType();

        return cast<DenseBitSequencesAttr>()
            .getType()
            .cast<BitSequenceLikeType>();
    }
    /// Gets the underlying BitSequenceType.
    [[nodiscard]] BitSequenceType getElementType() const
    {
        if (const auto bits = dyn_cast<BitSequenceAttr>())
            return bits.getType();

        return cast<DenseBitSequencesAttr>().getElementType();
    }

    /// Obtains a copy of this attribute bit casted to @p elementTy .
    ///
    /// @pre    `elementTy`
    /// @pre    `elementTy.getBitWidth() == getElementType().getBitWidth()`
    [[nodiscard]] BitSequenceLikeAttr
    bitCastElements(BitSequenceType elementTy) const
    {
        assert(elementTy);

        if (const auto bits = dyn_cast<BitSequenceAttr>())
            return bits.bitCast(elementTy);

        return cast<DenseBitSequencesAttr>()
            .bitCastElements(elementTy)
            .cast<BitSequenceLikeAttr>();
    }

    /// Applies @p fn to all values and returns the result.
    ///
    /// If @p elementTy is nullptr, getElementType() is used.
    ///
    /// @pre    bit width of @p elementTy and @p fn result matches
    [[nodiscard]] BitSequenceLikeAttr
    map(UnaryBitSequenceFn fn,
        BitSequenceType elementTy = {},
        bool allowSplat = true) const
    {
        // Handle single element case.
        if (const auto single = dyn_cast<BitSequenceAttr>())
            return single.map(fn, elementTy);

        // Handle dense case.
        return cast<DenseBitSequencesAttr>().map(fn, elementTy, allowSplat);
    }
    /// Combines the values with @p rhs using @p fn and return the result.
    ///
    /// If @p elementTy is nullptr, getElementType() is used.
    ///
    /// @pre    `rhs`
    /// @pre    shapes are compatible
    /// @pre    bit width of @p elementTy and @p fn result matches
    [[nodiscard]] BitSequenceLikeAttr
    zip(BinaryBitSequenceFn fn,
        BitSequenceLikeAttr rhs,
        BitSequenceType elementTy = {},
        bool allowSplat = true) const
    {
        assert(rhs);

        // Handle single element case.
        const auto singleLhs = dyn_cast<BitSequenceAttr>();
        const auto singleRhs = rhs.dyn_cast<BitSequenceAttr>();
        assert(static_cast<bool>(singleLhs) == static_cast<bool>(singleRhs));
        if (singleLhs) return singleLhs.zip(fn, singleRhs, elementTy);

        // Handle dense case.
        const auto denseLhs = cast<DenseBitSequencesAttr>();
        const auto denseRhs = rhs.cast<DenseBitSequencesAttr>();
        return denseLhs.zip(fn, denseRhs, elementTy, allowSplat);
    }
    /// Combines the values contained in this, @p arg1 and @p arg2 using @p fn .
    ///
    /// If @p elementTy is nullptr, getType() is used.
    ///
    /// @pre    `arg1 && arg2`
    /// @pre    shapes are compatible
    /// @pre    bit width of @p elementTy and @p fn result matches
    [[nodiscard]] BitSequenceLikeAttr
    zip(TernaryBitSequenceFn fn,
        BitSequenceLikeAttr arg1,
        BitSequenceLikeAttr arg2,
        BitSequenceType elementTy = {},
        bool allowSplat = true) const
    {
        assert(arg1 && arg2);

        // Handle single element case.
        const auto single0 = dyn_cast<BitSequenceAttr>();
        const auto single1 = arg1.dyn_cast<BitSequenceAttr>();
        const auto single2 = arg2.dyn_cast<BitSequenceAttr>();
        assert(static_cast<bool>(single0) == static_cast<bool>(single1));
        assert(static_cast<bool>(single0) == static_cast<bool>(single2));
        if (single0) return single0.zip(fn, single1, single2, elementTy);

        // Handle dense case.
        const auto dense0 = cast<DenseBitSequencesAttr>();
        const auto dense1 = arg1.cast<DenseBitSequencesAttr>();
        const auto dense2 = arg2.cast<DenseBitSequencesAttr>();
        return dense0.zip(fn, dense1, dense2, elementTy, allowSplat);
    }
};

/// Implements the BitSequenceAttr interface for built-in types.
void registerBitSequenceAttrModels(MLIRContext &ctx);

} // namespace mlir::bit
