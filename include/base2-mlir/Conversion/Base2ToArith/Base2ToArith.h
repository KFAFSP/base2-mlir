/// Declaration of the Base2 to SoftFloat dialect loowering pass.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

void populateBase2ToArithConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns,
    PatternBenefit benefit);

std::unique_ptr<Pass> createConvertBase2ToArithPass();

} // namespace mlir
