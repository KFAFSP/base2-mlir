/// Declaration of the SoftFLoat to libsoftfloat loowering pass.
///
/// @file
/// @author     Jihaong Bi (jiahong.bi@mailbox.tu-dresden.de)

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

void populateSoftFloatToLibConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns,
    PatternBenefit benefit);

std::unique_ptr<Pass> createConvertSoftFloatToLibPass();

} // namespace mlir
