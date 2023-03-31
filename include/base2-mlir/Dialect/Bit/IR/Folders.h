/// Declares the Bit dialect folding helper.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "base2-mlir/Dialect/Bit/IR/Ops.h"

namespace mlir::bit {

/// Singleton that implements Bit dialect operation folding.
class BitFolder {
public:
    [[nodiscard]] static OpFoldResult
    bitCast(CastOp op, ArrayRef<Attribute> operands);
    [[nodiscard]] static OpFoldResult
    bitCmp(CmpOp op, ArrayRef<Attribute> operands);
    [[nodiscard]] static OpFoldResult
    bitSelect(SelectOp op, ArrayRef<Attribute> operands);
    [[nodiscard]] static OpFoldResult
    bitAnd(AndOp op, ArrayRef<Attribute> operands);
    [[nodiscard]] static OpFoldResult
    bitOr(OrOp op, ArrayRef<Attribute> operands);
    [[nodiscard]] static OpFoldResult
    bitXor(XorOp op, ArrayRef<Attribute> operands);
    [[nodiscard]] static OpFoldResult
    bitShl(ShlOp op, ArrayRef<Attribute> operands);
    [[nodiscard]] static OpFoldResult
    bitShr(ShrOp op, ArrayRef<Attribute> operands);
    [[nodiscard]] static OpFoldResult
    bitCount(CountOp op, ArrayRef<Attribute> operands);
    [[nodiscard]] static OpFoldResult
    bitClz(ClzOp op, ArrayRef<Attribute> operands);
    [[nodiscard]] static OpFoldResult
    bitCtz(CtzOp op, ArrayRef<Attribute> operands);
};

} // namespace mlir::bit
