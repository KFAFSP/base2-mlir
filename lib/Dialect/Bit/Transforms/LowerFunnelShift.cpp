/// Implements the LowerFunnelShiftPass.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "base2-mlir/Dialect/Bit/IR/Bit.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::bit;

//===- Generated includes -------------------------------------------------===//

namespace mlir::bit {

#define GEN_PASS_DEF_LOWERFUNNELSHIFT
#include "base2-mlir/Dialect/Bit/Transforms/Passes.h.inc"

} // namespace mlir::bit

//===----------------------------------------------------------------------===//

namespace {

struct LowerFunnelShiftPass
        : mlir::bit::impl::LowerFunnelShiftBase<LowerFunnelShiftPass> {
    using LowerFunnelShiftBase::LowerFunnelShiftBase;

    void runOnOperation() override;
};

struct LowerFunnelShift : OpInterfaceRewritePattern<ShiftOp> {
    using OpInterfaceRewritePattern<ShiftOp>::OpInterfaceRewritePattern;

    virtual LogicalResult
    matchAndRewrite(ShiftOp op, PatternRewriter &rewriter) const override
    {
        // Only applies to funnel shifts.
        if (!op.getFunnel()) return failure();

        // Construct the inverse shift amount.
        const auto bitWidth = op.getValue()
                                  .getType()
                                  .cast<BitSequenceLikeType>()
                                  .getElementType()
                                  .getBitWidth();
        const auto invAmount = rewriter
                                   .create<index::SubOp>(
                                       op.getLoc(),
                                       rewriter
                                           .create<index::ConstantOp>(
                                               op.getLoc(),
                                               rewriter.getIndexAttr(bitWidth))
                                           .getResult(),
                                       op.getAmount())
                                   .getResult();

        // Implement using shift and or.
        switch (op.getDirection()) {
        case ShiftDirection::Left:
            rewriter.replaceOpWithNewOp<OrOp>(
                op,
                rewriter
                    .create<ShlOp>(op.getLoc(), op.getValue(), op.getAmount())
                    .getResult(),
                rewriter.create<ShrOp>(op.getLoc(), op.getFunnel(), invAmount)
                    .getResult());
            break;
        case ShiftDirection::Right:
            rewriter.replaceOpWithNewOp<OrOp>(
                op,
                rewriter
                    .create<ShrOp>(op.getLoc(), op.getValue(), op.getAmount())
                    .getResult(),
                rewriter.create<ShlOp>(op.getLoc(), op.getFunnel(), invAmount)
                    .getResult());
            break;
        }

        return success();
    }
};

} // namespace

void LowerFunnelShiftPass::runOnOperation()
{
    RewritePatternSet patterns(&getContext());

    populateLowerFunnelShiftPatterns(patterns);

    if (failed(applyPatternsAndFoldGreedily(
            getOperation(),
            FrozenRewritePatternSet(std::move(patterns)))))
        return signalPassFailure();
}

void mlir::bit::populateLowerFunnelShiftPatterns(RewritePatternSet &patterns)
{
    patterns.add<LowerFunnelShift>(patterns.getContext());
}

std::unique_ptr<Pass> mlir::bit::createLowerFunnelShiftPass()
{
    return std::make_unique<LowerFunnelShiftPass>();
}
