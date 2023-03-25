/// Declares some shared utility functions.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir::ext {

/// Obtains a type with the same shape as @p type , using @p elementTy .
///
/// @pre    `type`
/// @pre    `elementTy`
[[nodiscard]] inline Type getSameShape(Type type, Type elementTy)
{
    if (const auto shapedTy = type.dyn_cast<ShapedType>())
        return shapedTy.cloneWith(std::nullopt, elementTy);

    return elementTy;
}

/// Obtains the I1 or container-of-I1 that matches the shape of @p type .
///
/// @pre    `type`
[[nodiscard]] inline Type getI1SameShape(Type type)
{
    return getSameShape(type, IntegerType::get(type.getContext(), 1));
}

/// Combines @p attr and @p value into an OpFoldResult.
///
/// @post   `result || (!attr && !value)`
[[nodiscard]] inline OpFoldResult combine(Attribute attr, Value value)
{
    if (attr) return attr;
    return value;
}

/// Obtains the Attribute ConstantLike definition value of @p value .
///
/// @pre    `value`
[[nodiscard]] inline Attribute getConstantValue(Value value)
{
    assert(value);

    const auto def = value.getDefiningOp();
    if (!def || !def->hasTrait<OpTrait::ConstantLike>()) return Attribute{};

    SmallVector<OpFoldResult> folded;
    const auto result = def->fold(std::nullopt, folded);
    assert(succeeded(result));
    assert(folded.size() == 1);

    return folded.front().dyn_cast<Attribute>();
}

} // namespace mlir::ext
