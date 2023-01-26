/// Implements the ConvertBase2ToArithPass.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "base2-mlir/Conversion/Base2ToArith/Base2ToArith.h"

#include "base2-mlir/Dialect/Base2/IR/Base2.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::base2;

//===- Generated includes -------------------------------------------------===//

namespace mlir {

#define GEN_PASS_DEF_CONVERTBASE2TOARITH
#include "base2-mlir/Conversion/Passes.h.inc"

} // namespace mlir

//===----------------------------------------------------------------------===//

namespace {

struct ConvertBase2ToArithPass
        : impl::ConvertBase2ToArithBase<ConvertBase2ToArithPass> {
    using ConvertBase2ToArithBase::ConvertBase2ToArithBase;

    void runOnOperation() override;
};

} // namespace

void ConvertBase2ToArithPass::runOnOperation() {}

void mlir::populateBase2ToArithConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns,
    PatternBenefit benefit)
{}

std::unique_ptr<Pass> mlir::createConvertBase2ToArithPass()
{
    return std::make_unique<ConvertBase2ToArithPass>();
}
