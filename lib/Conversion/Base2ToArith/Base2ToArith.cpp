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

/// Determines whether @p entity is signed.
[[nodiscard]] static bool isSigned(auto entity)
{
    return entity.getSignedness() == Signedness::Signed;
}

[[nodiscard]] static bit_width_t toLeastBits(auto bits)
{
    if (bits > 0 && bits <= 8) return 8;
    if (bits > 8 && bits <= 16) return 16;
    if (bits > 16 && bits <= 32) return 32;
    if (bits > 32 && bits <= 64)
        return 64;
    else
        return 64;
}

struct ConvertBitCast final : public OpConversionPattern<BitCastOp> {
public:
    using OpConversionPattern<BitCastOp>::OpConversionPattern;

    ConvertBitCast(
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
                 .isSignlessIntOrFloat())
            return rewriter.notifyMatchFailure(
                op,
                "expected signless int or float input");
        const auto resultTy = typeConverter->convertType(op.getType());
        if (!getElementTypeOrSelf(resultTy).isSignlessIntOrFloat())
            return rewriter.notifyMatchFailure(
                op,
                "expected signless int or float input");

        rewriter.replaceOpWithNewOp<arith::BitcastOp>(
            op,
            resultTy,
            adaptor.getIn());

        return success();
    }
};

struct ConvertExtOrTrunc : OpConversionPattern<ValueCastOp> {
    using OpConversionPattern::OpConversionPattern;

    virtual LogicalResult matchAndRewrite(
        ValueCastOp op,
        typename ValueCastOp::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // Only applies to known-exact operations.
        if (op.getRoundingMode() != RoundingMode::None)
            return rewriter.notifyMatchFailure(op, [&](auto &diag) {
                diag << "unsupported rounding mode `"
                     << stringifyRoundingMode(op.getRoundingMode()) << "`";
            });

        // Only applies to signless integer or float input and output.
        if (!adaptor.getIn().getType().isSignlessIntOrFloat())
            return rewriter.notifyMatchFailure(
                op,
                "expected signless int or float input");
        const auto resultTy = typeConverter->convertType(op.getType());
        if (!resultTy.isSignlessIntOrFloat())
            return rewriter.notifyMatchFailure(
                op,
                "expected signless int or float output");
        if (adaptor.getIn().getType().isa<IntegerType>()
            != resultTy.isa<IntegerType>())
            return rewriter.notifyMatchFailure(op, "expected matching types");

        // Determine the bit-widths.
        const auto inBits = adaptor.getIn().getType().getIntOrFloatBitWidth();
        const auto outBits = resultTy.getIntOrFloatBitWidth();

        // Handle truncation.
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
            if (adaptor.getIn().getType().isa<FloatType>()) {
                rewriter.replaceOpWithNewOp<arith::ExtFOp>(
                    op,
                    resultTy,
                    adaptor.getIn());
                return success();
            }

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

struct ConvertIntToFPCast : OpConversionPattern<ValueCastOp> {
    using OpConversionPattern::OpConversionPattern;

    virtual LogicalResult matchAndRewrite(
        ValueCastOp op,
        typename ValueCastOp::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // Only applies to known-exact operations.
        if (op.getRoundingMode() != RoundingMode::None)
            return rewriter.notifyMatchFailure(op, [&](auto &diag) {
                diag << "unsupported rounding mode `"
                     << stringifyRoundingMode(op.getRoundingMode()) << "`";
            });

        // Only applies to signless integer input and float output.
        if (!adaptor.getIn().getType().isSignlessInteger())
            return rewriter.notifyMatchFailure(
                op,
                "expected signless int input");
        const auto resultTy = typeConverter->convertType(op.getType());
        if (!resultTy.isa<FloatType>())
            return rewriter.notifyMatchFailure(op, "expected float output");

        // Handle signed.
        if (isSigned(op.getIn().getType().dyn_cast<FixedPointSemantics>())) {
            rewriter.replaceOpWithNewOp<arith::SIToFPOp>(
                op,
                resultTy,
                adaptor.getIn());
            return success();
        }

        // Handle unsigned.
        rewriter.replaceOpWithNewOp<arith::UIToFPOp>(
            op,
            resultTy,
            adaptor.getIn());
        return success();
    }
};

struct ConvertFPToIntCast : OpConversionPattern<ValueCastOp> {
    using OpConversionPattern::OpConversionPattern;

    virtual LogicalResult matchAndRewrite(
        ValueCastOp op,
        typename ValueCastOp::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // Only applies to known-exact operations.
        if (op.getRoundingMode() != RoundingMode::None)
            return rewriter.notifyMatchFailure(op, [&](auto &diag) {
                diag << "unsupported rounding mode `"
                     << stringifyRoundingMode(op.getRoundingMode()) << "`";
            });

        // Only applies to float input and signless integer output.
        if (!adaptor.getIn().getType().isa<FloatType>())
            return rewriter.notifyMatchFailure(op, "expected float input");
        const auto resultTy = typeConverter->convertType(op.getType());
        if (!resultTy.isSignlessInteger())
            return rewriter.notifyMatchFailure(
                op,
                "expected signless int output");

        // Handle signed.
        if (isSigned(op.getType().dyn_cast<FixedPointSemantics>())) {
            rewriter.replaceOpWithNewOp<arith::FPToSIOp>(
                op,
                resultTy,
                adaptor.getIn());
            return success();
        }

        // Handle unsigned.
        rewriter.replaceOpWithNewOp<arith::FPToUIOp>(
            op,
            resultTy,
            adaptor.getIn());
        return success();
    }
};

template<class From, class FP, class SI, class UI = SI>
struct ConvertExactBinOp : OpConversionPattern<From> {
    using OpConversionPattern<From>::OpConversionPattern;

    virtual LogicalResult matchAndRewrite(
        From op,
        typename From::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        if (op.getRoundingMode() != RoundingMode::None)
            return rewriter.notifyMatchFailure(op, [&](auto &diag) {
                diag << "unsupported rounding mode `"
                     << stringifyRoundingMode(op.getRoundingMode()) << "`";
            });

        if (adaptor.getLhs().getType().template isa<IntegerType>())
            return convertInt(op, adaptor, rewriter);
        if (adaptor.getLhs().getType().template isa<FloatType>())
            return convertFloat(op, adaptor, rewriter);

        return rewriter.notifyMatchFailure(
            op,
            "expected signless int or float");
    }

    LogicalResult convertInt(
        From op,
        typename From::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const
    {
        auto signedness = op.getLhs()
                              .getType()
                              .template dyn_cast<FixedPointSemantics>()
                              .getSignedness();
        if constexpr (std::is_same_v<SI, UI>) {
            // Ignore signedness if it doesn't matter so that we can
            // transparently access the sign-agnostic arith operations.
            if (signedness == Signedness::Signless)
                signedness = Signedness::Unsigned;
        }

        const auto inBits = adaptor.getLhs().getType().getIntOrFloatBitWidth();
        auto resultType = IntegerType::get(rewriter.getContext(), inBits);
        const auto upBits = toLeastBits(inBits);
        Value newLhs, newRhs;
        if (inBits != upBits) {
            newLhs = rewriter.create<ValueCastOp>(
                op.getLoc(),
                IntegerType::get(rewriter.getContext(), upBits),
                adaptor.getLhs());
            newRhs = rewriter.create<ValueCastOp>(
                op.getLoc(),
                IntegerType::get(rewriter.getContext(), upBits),
                adaptor.getRhs());
        }

        switch (signedness) {
        case Signedness::Unsigned:
            if (inBits != upBits) {
                auto result = rewriter.create<UI>(op.getLoc(), newLhs, newRhs);
                rewriter.replaceOpWithNewOp<ValueCastOp>(
                    op,
                    resultType,
                    result);
            } else {
                rewriter.replaceOpWithNewOp<UI>(
                    op,
                    adaptor.getLhs(),
                    adaptor.getRhs());
            }
            break;
        case Signedness::Signed:
            if (inBits != upBits) {
                auto result = rewriter.create<SI>(op.getLoc(), newLhs, newRhs);
                rewriter.replaceOpWithNewOp<ValueCastOp>(
                    op,
                    resultType,
                    result);
            } else {
                rewriter.replaceOpWithNewOp<SI>(
                    op,
                    adaptor.getLhs(),
                    adaptor.getRhs());
            }
            break;
        case Signedness::Signless:
            return rewriter.notifyMatchFailure(op, "undefined semantics");
        }

        return success();
    }

    LogicalResult convertFloat(
        From op,
        typename From::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const
    {
        rewriter.replaceOpWithNewOp<FP>(op, adaptor.getLhs(), adaptor.getRhs());
        return success();
    }
};

using ConvertExactAdd = ConvertExactBinOp<AddOp, arith::AddFOp, arith::AddIOp>;
using ConvertExactSub = ConvertExactBinOp<SubOp, arith::SubFOp, arith::SubIOp>;
using ConvertExactMul = ConvertExactBinOp<MulOp, arith::MulFOp, arith::MulIOp>;

} // namespace

void mlir::populateBase2ToArithConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns,
    PatternBenefit benefit)
{
    patterns.add<
        ConvertBitCast,
        ConvertExtOrTrunc,
        ConvertIntToFPCast,
        ConvertFPToIntCast,
        ConvertExactAdd,
        ConvertExactSub,
        ConvertExactMul>(typeConverter, patterns.getContext(), benefit);
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
        if (!sema.getFractionalBits()) return type;

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

    // target.addDynamicallyLegalDialect<Base2Dialect>([&](Operation* op) {
    //     return converter.isSignatureLegal(getOpFunctionType(op));
    // });
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
