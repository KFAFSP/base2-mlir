/// Implements the LowerBitwiseLogicPass.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "base2-mlir/Dialect/Bit/IR/Bit.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <memory>

using namespace mlir;
using namespace mlir::bit;

/// Obtains a type with the same shape as @p type , using @p elementTy .
///
/// @pre    `type`
/// @pre    `elementTy`
[[nodiscard]] static Type getSameShape(Type type, Type elementTy)
{
    if (const auto shapedTy = type.dyn_cast<ShapedType>())
        return shapedTy.cloneWith(std::nullopt, elementTy);

    return elementTy;
}

[[nodiscard]] static Type getSignlessType(BitSequenceLikeType type)
{
    const auto bitWidth = type.getElementType().getBitWidth();
    const auto elementTy = IntegerType::get(type.getContext(), bitWidth);
    return getSameShape(type, elementTy);
}

//===- Generated includes -------------------------------------------------===//

namespace mlir::bit {

#define GEN_PASS_DEF_LOWERBITWISELOGIC
#include "base2-mlir/Dialect/Bit/Transforms/Passes.h.inc"

} // namespace mlir::bit

//===----------------------------------------------------------------------===//

namespace {

struct LowerBitwiseLogicPass
        : mlir::bit::impl::LowerBitwiseLogicBase<LowerBitwiseLogicPass> {
    using LowerBitwiseLogicBase::LowerBitwiseLogicBase;

    void runOnOperation() override;
};

template<class Op>
struct LowerBinaryOp : OpRewritePattern<Op> {
    using OpRewritePattern<Op>::OpRewritePattern;

    virtual LogicalResult
    matchAndRewrite(Op op, PatternRewriter &rewriter) const override
    {
        const auto lhsTy =
            op.getLhs().getType().template cast<BitSequenceLikeType>();
        const auto rhsTy =
            op.getRhs().getType().template cast<BitSequenceLikeType>();
        if (lhsTy.getElementType().isSignlessInteger()
            && rhsTy.getElementType().isSignlessInteger())
            return failure();

        auto resultOp = rewriter.create<Op>(
            op.getLoc(),
            rewriter
                .create<CastOp>(
                    op.getLoc(),
                    getSignlessType(lhsTy),
                    op.getLhs())
                .getResult(),
            rewriter
                .create<CastOp>(
                    op.getLoc(),
                    getSignlessType(rhsTy),
                    op.getRhs())
                .getResult());
        resultOp->setAttrs(op->getAttrs());

        const auto resultTy = cast<BitSequenceLikeType>(op.getType());
        if (resultTy.getElementType().isSignlessInteger())
            rewriter.replaceOp(op, resultOp.getResult());
        else
            rewriter.replaceOpWithNewOp<CastOp>(
                op,
                op.getType(),
                resultOp.getResult());

        return success();
    }
};

using LowerAndOp = LowerBinaryOp<AndOp>;
using LowerOrOp = LowerBinaryOp<OrOp>;
using LowerXorOp = LowerBinaryOp<XorOp>;

} // namespace

void LowerBitwiseLogicPass::runOnOperation()
{
    RewritePatternSet patterns(&getContext());

    populateLowerBitwiseLogicPatterns(patterns);

    if (failed(applyPatternsAndFoldGreedily(
            getOperation(),
            FrozenRewritePatternSet(std::move(patterns)))))
        return signalPassFailure();
}

void mlir::bit::populateLowerBitwiseLogicPatterns(RewritePatternSet &patterns)
{
    patterns.add<LowerAndOp, LowerOrOp, LowerXorOp>(patterns.getContext());
}

std::unique_ptr<Pass> mlir::bit::createLowerBitwiseLogicPass()
{
    return std::make_unique<LowerBitwiseLogicPass>();
}
