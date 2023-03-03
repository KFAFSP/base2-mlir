/// Implements the Base2ToArith pass.
///
/// @file
/// @author     Jihaong Bi (jiahong.bi@mailbox.tu-dresden.de)
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "base2-mlir/Conversion/Base2ToArith/Base2ToArith.h"

#include "../PassDetails.h"
#include "base2-mlir/Dialect/Base2/IR/Base2.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace mlir::base2;

namespace {

[[nodiscard]] static bool isSigned(auto entity)
{
    return entity.getSignedness() == Signedness::Signed;
}

struct ConvertBitcast : OpConversionPattern<BitCastOp> {
    using OpConversionPattern<BitCastOp>::OpConversionPattern;

    ConvertBitcast(
        TypeConverter &typeConverter,
        MLIRContext* context,
        PatternBenefit benefit)
            : OpConversionPattern<BitCastOp>(typeConverter, context, benefit){};

    LogicalResult matchAndRewrite(
        BitCastOp op,
        BitCastOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        assert(adaptor.getOperands().size() == 1);

        if (!getElementTypeOrSelf(adaptor.getIn().getType())
                 .isSignlessInteger())
            return rewriter.notifyMatchFailure(op, "expected signless integer");

        const auto resultTy = typeConverter->convertType(op.getType());
        if (!getElementTypeOrSelf(resultTy).isSignlessInteger())
            return rewriter.notifyMatchFailure(op, "epected signless integer");

        rewriter.replaceOpWithNewOp<arith::BitcastOp>(
            op,
            resultTy,
            adaptor.getIn());

        return success();
    }
};

struct ConvertExtOrTrunc : OpConversionPattern<ValueCastOp> {
    using OpConversionPattern<ValueCastOp>::OpConversionPattern;

    ConvertExtOrTrunc(
        TypeConverter &typeConverter,
        MLIRContext* context,
        PatternBenefit benefit)
            : OpConversionPattern<ValueCastOp>(
                typeConverter,
                context,
                benefit){};

    LogicalResult matchAndRewrite(
        ValueCastOp op,
        ValueCastOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        /// Test with no rounding first
        /// TODO: Rounding using elaborator
        if (op.getRoundingMode() != RoundingMode::None)
            return rewriter.notifyMatchFailure(op, [&](auto &diag) {
                diag << "unsupported rounding mode `"
                     << stringifyRoundingMode(op.getRoundingMode()) << "`";
            });

        if (!adaptor.getIn().getType().isSignlessIntOrFloat())
            return rewriter.notifyMatchFailure(op, "expected signless integer");
        const auto resultTy = typeConverter->convertType(op.getType());
        if (!resultTy.isSignlessIntOrFloat())
            return rewriter.notifyMatchFailure(
                op,
                "expected signless int or float output");
        if (adaptor.getIn().getType().isa<IntegerType>()
            != resultTy.isa<IntegerType>())
            return rewriter.notifyMatchFailure(op, "expected matching types");

        const auto inBits = adaptor.getIn().getType().getIntOrFloatBitWidth();
        const auto outBits = resultTy.getIntOrFloatBitWidth();

        if (inBits > outBits) {
            if (adaptor.getIn().getType().isa<FloatType>()) {
                rewriter.replaceOpWithNewOp<arith::TruncFOp>(
                    op,
                    resultTy,
                    adaptor.getIn());
                return success();
            }

            rewriter.replaceOpWithNewOp<arith::TruncIOp>(
                op,
                resultTy,
                adaptor.getIn());
            return success();
        }

        // Handle extension.
        if (inBits < outBits) {
            // Handle signed.
            if (isSigned(
                    op.getIn().getType().dyn_cast<FixedPointSemantics>())) {
                rewriter.replaceOpWithNewOp<arith::ExtSIOp>(
                    op,
                    resultTy,
                    adaptor.getIn());
                return success();
            }

            // Handle unsigned.
            rewriter.replaceOpWithNewOp<arith::ExtUIOp>(
                op,
                resultTy,
                adaptor.getIn());
            return success();
        }

        // Turns out this is just a bitcast.
        rewriter.replaceOp(op, adaptor.getIn());

        return success();
    }
};

struct ConvertFixedAddOp : OpConversionPattern<FixedAddOp> {
    using OpConversionPattern<FixedAddOp>::OpConversionPattern;

    ConvertFixedAddOp(
        TypeConverter &typeConverter,
        MLIRContext* context,
        PatternBenefit benefit)
            : OpConversionPattern<FixedAddOp>(
                typeConverter,
                context,
                benefit){};

    LogicalResult matchAndRewrite(
        FixedAddOp op,
        FixedAddOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        assert(adaptor.getOperands().size() == 2);

        const auto lhsTy = adaptor.getLhs().getType();
        const auto rhsTy = adaptor.getRhs().getType();
        assert(lhsTy.isSignlessInteger() && rhsTy.isSignlessInteger());
        const auto resultTy = typeConverter->convertType(op.getType());

        Value newLhs =
            rewriter.create<ValueCastOp>(op.getLoc(), resultTy, op.getLhs());
        Value newRhs =
            rewriter.create<ValueCastOp>(op.getLoc(), resultTy, op.getRhs());

        rewriter.replaceOpWithNewOp<arith::AddIOp>(op, newLhs, newRhs);

        return success();
    }
};

} // namespace

void mlir::populateBase2ToArithConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns,
    PatternBenefit benefit)
{
    patterns.add<ConvertBitcast, ConvertExtOrTrunc, ConvertFixedAddOp>(
        typeConverter,
        patterns.getContext(),
        benefit);
}

namespace {
struct ConvertBase2ToArithPass
        : public ConvertBase2ToArithBase<ConvertBase2ToArithPass> {
    void runOnOperation() final;
};
} // namespace

void ConvertBase2ToArithPass::runOnOperation()
{
    TypeConverter converter;

    // Allow all unknown types.
    converter.addConversion([&](Type type) { return type; });
    // Convert integer fixed-point types to signless built-in integers.
    converter.addConversion([](FixedPointLikeType type) -> Type {
        const auto sema = type.getSemantics();
        if (!sema.getExponent()) return type;

        return IntegerType::get(sema.getContext(), sema.getBitWidth());
    });
    // Convert the element types of containers.
    converter.addConversion([&](ShapedType shapedTy) -> Type {
        return shapedTy.cloneWith(
            std::nullopt,
            converter.convertType(shapedTy.getElementType()));
    });
    // Materialization can be inserted using bitcast.
    converter.addSourceMaterialization(
        [](OpBuilder &builder, Type resultTy, ValueRange inputs, Location loc)
            -> Optional<Value> {
            if (getElementTypeOrSelf(resultTy).isa<IntegerType, FloatType>()) {
                assert(inputs.size() == 1);
                return builder.create<BitCastOp>(loc, resultTy, inputs.front())
                    .getResult();
            }

            return std::nullopt;
        });
    converter.addSourceMaterialization(
        [](OpBuilder &builder, Type resultTy, ValueRange inputs, Location loc)
            -> Optional<Value> {
            if (getElementTypeOrSelf(resultTy).isa<FixedPointLikeType>()) {
                assert(inputs.size() == 1);
                return builder.create<BitCastOp>(loc, resultTy, inputs.front())
                    .getResult();
            }

            return std::nullopt;
        });

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    // Convert function signatures.
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns,
        converter);
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
        return converter.isSignatureLegal(op.getFunctionType())
               && converter.isLegal(&op.getBody());
    });
    populateCallOpTypeConversionPattern(patterns, converter);
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](func::CallOp op) { return converter.isLegal(op); });
    populateReturnOpTypeConversionPattern(patterns, converter);
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](func::ReturnOp op) { return converter.isLegal(op); });

    // Convert base2 dialect operations
    populateBase2ToArithConversionPatterns(converter, patterns, 1);

    target.addIllegalDialect<Base2Dialect>();
    target.addLegalDialect<BuiltinDialect, arith::ArithDialect>();

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns))))
        signalPassFailure();
}

std::unique_ptr<Pass> mlir::createConvertBase2ToArithPass()
{
    return std::make_unique<ConvertBase2ToArithPass>();
}
