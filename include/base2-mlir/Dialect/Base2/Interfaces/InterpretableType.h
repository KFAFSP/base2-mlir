/// Declares the Base2 InterpretableType interface.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "base2-mlir/Dialect/Base2/Analysis/ValueFacts.h"
#include "base2-mlir/Dialect/Base2/Enums.h"
#include "base2-mlir/Dialect/Bit/Interfaces/BitSequenceType.h"
#include "mlir/IR/OpDefinition.h"

#include <cassert>
#include <compare>
#include <optional>

namespace mlir::base2 {

/// Optional bit::BitSequence result type.
using bit_result = std::optional<bit::BitSequence>;
/// Optional std::partial_ordering result type.
using cmp_result = std::optional<std::partial_ordering>;

} // namespace mlir::base2

namespace mlir::base2::interpretable_type_interface_defaults {

/// Attempts a value cast on zeros to determine legality.
[[nodiscard]] bool canValueCast(Type self, Type from, Type to);

/// Compares and selects using cmp.
[[nodiscard]] std::optional<bit::BitSequence>
min(Type self, const bit::BitSequence &lhs, const bit::BitSequence &rhs);

/// Compares and selects using cmp.
[[nodiscard]] std::optional<bit::BitSequence>
max(Type self, const bit::BitSequence &lhs, const bit::BitSequence &rhs);

/// Checks trivial equality.
[[nodiscard]] std::optional<std::partial_ordering>
cmp(Type self, const bit::BitSequence &lhs, const bit::BitSequence &rhs);

} // namespace mlir::base2::interpretable_type_interface_defaults

//===- Generated includes -------------------------------------------------===//

#include "base2-mlir/Dialect/Base2/Interfaces/InterpretableType.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::base2 {

//===----------------------------------------------------------------------===//
// InterpretableLikeType
//===----------------------------------------------------------------------===//

/// Concept for a type or container type of interpretable bit sequences.
///
/// Satisfied by either an InterpretableType or a ShapedType with
/// InterpretableType elements.
class InterpretableLikeType : public Type {
public:
    using Type::Type;

    /// @copydoc classof(Type)
    [[nodiscard]] static bool classof(InterpretableType) { return true; }
    /// @copydoc classof(Type)
    [[nodiscard]] static bool classof(ShapedType type)
    {
        return type.getElementType().isa<InterpretableType>();
    }
    /// Determines whether @p type is a InterpretableLikeType.
    [[nodiscard]] static bool classof(Type type)
    {
        if (type.isa<InterpretableType>()) return true;
        if (const auto shapedTy = type.dyn_cast<ShapedType>())
            return classof(shapedTy);

        return false;
    }

    /// Initializes an InterpretableLikeType from @p type .
    ///
    /// @pre    `type`
    /*implicit*/ InterpretableLikeType(InterpretableType type)
            : Type(type.cast<Type>().getImpl())
    {}

    /// Gets the underlying InterpretableType.
    [[nodiscard]] InterpretableType getElementType() const
    {
        if (const auto self = dyn_cast<InterpretableType>()) return self;

        return cast<ShapedType>().getElementType().cast<InterpretableType>();
    }

    /// Gets a InterpretableLikeType of the same shape with @p elementTy .
    ///
    /// @pre    `elementTy`
    [[nodiscard]] InterpretableLikeType
    getSameShape(InterpretableType elementTy) const
    {
        assert(elementTy);

        if (isa<InterpretableType>()) return elementTy;

        return cast<ShapedType>()
            .cloneWith(std::nullopt, elementTy)
            .cast<InterpretableLikeType>();
    }
};

/// Implements the InterpretableType interface for built-in types.
void registerInterpretableTypeModels(MLIRContext &ctx);

} // namespace mlir::base2
