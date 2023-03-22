/// Declares the Bit BitSequenceType interface.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "base2-mlir/Dialect/Bit/Analysis/BitSequence.h"
#include "mlir/IR/OpDefinition.h"

#include <cassert>

//===- Generated includes -------------------------------------------------===//

#include "base2-mlir/Dialect/Bit/Interfaces/BitSequenceType.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::bit {

//===----------------------------------------------------------------------===//
// BitSequenceLikeType
//===----------------------------------------------------------------------===//

/// Concept for a type or container type of bit sequences.
///
/// Satisfied by either a BitSequenceType or a ShapedType with BitSequenceType
/// elements.
class BitSequenceLikeType : public Type {
public:
    using Type::Type;

    /// @copydoc classof(Type)
    [[nodiscard]] static bool classof(BitSequenceType) { return true; }
    /// @copydoc classof(Type)
    [[nodiscard]] static bool classof(ShapedType type)
    {
        return type.getElementType().isa<BitSequenceType>();
    }
    /// Determines whether @p type is a BitSequenceLikeType.
    [[nodiscard]] static bool classof(Type type)
    {
        if (type.isa<BitSequenceType>()) return true;
        if (const auto shapedTy = type.dyn_cast<ShapedType>())
            return classof(shapedTy);

        return false;
    }

    /// Initializes a BitSequenceLikeType from @p type .
    ///
    /// @pre    `type`
    /*implicit*/ BitSequenceLikeType(BitSequenceType type)
            : Type(type.cast<Type>().getImpl())
    {}

    /// Gets the underlying BitSequenceType.
    [[nodiscard]] BitSequenceType getElementType() const
    {
        if (const auto self = dyn_cast<BitSequenceType>()) return self;

        return cast<ShapedType>().getElementType().cast<BitSequenceType>();
    }

    /// Gets a BitSequenceLikeType of the same shape with @p elementTy .
    ///
    /// If this is not a shaped type, returns @p elementTy .
    ///
    /// @pre    `elementTy`
    [[nodiscard]] BitSequenceLikeType
    getSameShape(BitSequenceType elementTy) const
    {
        assert(elementTy);

        if (isa<BitSequenceType>()) return elementTy;

        return cast<ShapedType>()
            .cloneWith(std::nullopt, elementTy)
            .cast<BitSequenceLikeType>();
    }
};

/// Implements the BitSequenceType interface for built-in types.
void registerBitSequenceTypeModels(MLIRContext &ctx);

} // namespace mlir::bit
