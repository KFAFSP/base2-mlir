/// Implements the LowerBitwisePass.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "base2-mlir/Dialect/Bit/IR/Bit.h"
#include "base2-mlir/Utils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::bit;
using namespace mlir::ext;

/// Obtains a (container of) signless type for width and shape of @p type .
///
/// @pre    `type`
[[nodiscard]] static Type getSignlessType(BitSequenceLikeType type)
{
    const auto bitWidth = type.getElementType().getBitWidth();
    const auto elementTy = IntegerType::get(type.getContext(), bitWidth);
    return getSameShape(type, elementTy);
}

//===- Generated includes -------------------------------------------------===//

namespace mlir::bit {

#define GEN_PASS_DEF_LOWERBITWISE
#include "base2-mlir/Dialect/Bit/Transforms/Passes.h.inc"

} // namespace mlir::bit

//===----------------------------------------------------------------------===//

namespace {

struct LowerBitwisePass : mlir::bit::impl::LowerBitwiseBase<LowerBitwisePass> {
    using LowerBitwiseBase::LowerBitwiseBase;

    void runOnOperation() override;
};

LogicalResult matchAndRewrite(Operation* op, PatternRewriter &rewriter)
{
    assert(op);
    assert(op->getNumResults() == 1);
    assert(op->getNumSuccessors() == 0);
    assert(op->getNumRegions() == 0);

    // Prepare the replacement operation state.
    OperationState cloneState(
        op->getLoc(),
        op->getName(),
        op->getOperands(),
        op->getResultTypes(),
        op->getAttrs());

    // Process all operands.
    auto matched = false;
    for (auto [idx, opd] : llvm::enumerate(cloneState.operands)) {
        // Ignore non bit sequence operands.
        const auto opdTy = opd.getType().dyn_cast<BitSequenceLikeType>();
        if (!opdTy) continue;
        // Ignore already signless operands.
        const auto signlessTy = getSignlessType(opdTy);
        if (opdTy == signlessTy) continue;

        // Cast to signless type.
        matched |= true;
        cloneState.operands[idx] =
            rewriter.create<CastOp>(op->getLoc(), signlessTy, opd).getResult();
    }

    if (!matched) {
        // Nothing was rewritten.
        return failure();
    }

    // Process result type.
    if (const auto resTy =
            cloneState.types.front().dyn_cast<BitSequenceLikeType>())
        cloneState.types.front() = getSignlessType(resTy);

    // Create the cloned op.
    auto result = rewriter.create(cloneState)->getResult(0);

    // Cast the result if necessary.
    if (result.getType() != op->getResultTypes()[0]) {
        rewriter.replaceOpWithNewOp<CastOp>(
            op,
            op->getResultTypes()[0],
            result);
    } else {
        rewriter.replaceOp(op, result);
    }

    return success();
}

template<class Op>
struct LowerSimpleOp : OpRewritePattern<Op> {
    using OpRewritePattern<Op>::OpRewritePattern;

    virtual LogicalResult
    matchAndRewrite(Op op, PatternRewriter &rewriter) const override
    {
        return ::matchAndRewrite(op, rewriter);
    }
};

using LowerCmpOp = LowerSimpleOp<CmpOp>;
using LowerAndOp = LowerSimpleOp<AndOp>;
using LowerOrOp = LowerSimpleOp<OrOp>;
using LowerXorOp = LowerSimpleOp<XorOp>;
using LowerShlOp = LowerSimpleOp<ShlOp>;
using LowerShrOp = LowerSimpleOp<ShrOp>;
using LowerCountOp = LowerSimpleOp<CountOp>;
using LowerClzOp = LowerSimpleOp<ClzOp>;
using LowerCtzOp = LowerSimpleOp<CtzOp>;

} // namespace

void LowerBitwisePass::runOnOperation()
{
    RewritePatternSet patterns(&getContext());

    populateLowerBitwisePatterns(patterns);

    if (failed(applyPatternsAndFoldGreedily(
            getOperation(),
            FrozenRewritePatternSet(std::move(patterns)))))
        return signalPassFailure();
}

void mlir::bit::populateLowerBitwisePatterns(RewritePatternSet &patterns)
{
    patterns.add<
        LowerCmpOp,
        LowerAndOp,
        LowerOrOp,
        LowerXorOp,
        LowerShlOp,
        LowerShrOp,
        LowerCountOp,
        LowerClzOp,
        LowerCtzOp>(patterns.getContext());
}

std::unique_ptr<Pass> mlir::bit::createLowerBitwisePass()
{
    return std::make_unique<LowerBitwisePass>();
}
