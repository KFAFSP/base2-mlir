/// Declares the Bit passes.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL
#include "base2-mlir/Dialect/Bit/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

namespace bit {

/// Adds the lower-bitwise-logic pass patterns to @p patterns .
void populateLowerBitwiseLogicPatterns(RewritePatternSet &patterns);

/// Constructs the lower-bitwise-logic pass.
std::unique_ptr<Pass> createLowerBitwiseLogicPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "base2-mlir/Dialect/Bit/Transforms/Passes.h.inc"

} // namespace bit

} // namespace mlir