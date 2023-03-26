/// Declaration of the Bit to LLVM conversion pass.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL_CONVERTBITTOLLVM
#include "base2-mlir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//

namespace bit {

/// Adds the convert-bit-to-llvm pass patterns to @p patterns .
void populateConvertBitToLLVMPatterns(
    LLVMTypeConverter &converter,
    RewritePatternSet &patterns);

} // namespace bit

/// Constructs the convert-bit-to-llvm pass.
std::unique_ptr<Pass> createConvertBitToLLVMPass();

} // namespace mlir
