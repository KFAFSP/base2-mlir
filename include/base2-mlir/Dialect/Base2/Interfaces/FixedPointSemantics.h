/// Declares the Base2 FixedPointSemantics interface.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "base2-mlir/Dialect/Base2/Analysis/BitSequence.h"
#include "base2-mlir/Dialect/Base2/Enums.h"
#include "base2-mlir/Dialect/Base2/Interfaces/BitSequenceType.h"
#include "base2-mlir/Dialect/Base2/Interfaces/InterpretableType.h"
#include "mlir/IR/OpDefinition.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

//===- Generated includes -------------------------------------------------===//

#include "base2-mlir/Dialect/Base2/Interfaces/FixedPointSemantics.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::base2 {

//===----------------------------------------------------------------------===//
// FixedPointLikeType
//===----------------------------------------------------------------------===//

/// Concept for a type or container type of fixed-point numbers.
///
/// Satisfied by a BitSequenceLikeType, the elements of which also satisfy
/// FixedPointSemantics.
class FixedPointLikeType : public BitSequenceLikeType {
public:
    /// @copydoc classof(Type)
    [[nodiscard]] static bool classof(BitSequenceLikeType type)
    {
        return type.getElementType().isa<FixedPointSemantics>();
    }
    /// Determines whether @p type is a FixedPointLikeType.
    [[nodiscard]] static bool classof(Type type)
    {
        if (const auto bitSequenceLikeTy = type.dyn_cast<BitSequenceLikeType>())
            return classof(bitSequenceLikeTy);

        return false;
    }

    using BitSequenceLikeType::BitSequenceLikeType;
    /*implicit*/ FixedPointLikeType(BitSequenceType) = delete;
    /// Initializes a FixedPointLikeType from @p type .
    ///
    /// @pre    `type`
    /*implicit*/ FixedPointLikeType(FixedPointSemantics type)
            : BitSequenceLikeType(type.cast<Type>().getImpl())
    {}

    /// Gets the underlying FixedPointSemantics.
    [[nodiscard]] FixedPointSemantics getSemantics() const
    {
        if (const auto self = dyn_cast<FixedPointSemantics>()) return self;

        return getElementType().cast<FixedPointSemantics>();
    }
};

/// Implements the FixedPointSemantics interface for built-in types.
void registerFixedPointSemanticsModels(MLIRContext &ctx);

} // namespace mlir::base2
