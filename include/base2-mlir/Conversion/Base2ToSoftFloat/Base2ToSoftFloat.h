/// Declaration of the Base2 to SoftFloat dialect loowering pass.
///
/// @file
/// @author     Jihaong Bi (jiahong.bi@mailbox.tu-dresden.de)

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

void populateBase2ToSoftFloatConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns,
    PatternBenefit benefit);

std::unique_ptr<Pass> createConvertBase2ToSoftFloatPass();

} // namespace mlir
