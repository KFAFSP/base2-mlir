/// Implements the Base2ToSoftFloat pass.
///
/// @file
/// @author     Jihaong Bi (jiahong.bi@mailbox.tu-dresden.de)

#include "base2-mlir/Conversion/Base2ToSoftFloat/Base2ToSoftFloat.h"

#include "../PassDetails.h"
#include "base2-mlir/Dialect/Base2/IR/Base2.h"
#include "base2-mlir/Dialect/SoftFloat/IR/SoftFloat.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::base2;

namespace {

static std::optional<int> toLibRounding(RoundingMode roundingMode)
{
    switch (roundingMode) {
    case RoundingMode::None: return 0;
    case RoundingMode::Nearest: return 1;
    default: return std::nullopt;
    }
}

static std::optional<int> toNanBit(PartialOrderingPredicate predicate)
{
    if (predicate > PartialOrderingPredicate::Falsum
        && predicate < PartialOrderingPredicate::Ordered)
        return 0;
    else if (
        predicate > PartialOrderingPredicate::Unordered
        && predicate < PartialOrderingPredicate::Verum)
        return 1;
    else
        return std::nullopt;
}

// pattern to convert arithmetic operations of type ieee754 to softfloat
// dialect
template<typename Op>
struct ClosedArithOpLowering final : public OpConversionPattern<Op> {
public:
    using OpConversionPattern<Op>::OpConversionPattern;

    ClosedArithOpLowering<Op>(
        TypeConverter &typeConverter,
        MLIRContext* context,
        StringRef function,
        PatternBenefit benefit)
            : OpConversionPattern<Op>(typeConverter, context, benefit),
              function(function){};

    LogicalResult matchAndRewrite(
        Op op,
        typename Op::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        assert(adaptor.getOperands().size() <= 3);

        if (op.getResult().getType().template isa<ShapedType>())
            return rewriter.notifyMatchFailure(op, "expected scalar operation");

        Location loc = op->getLoc();

        TypeConverter* converter = this->typeConverter;
        auto type = getElementTypeOrSelf(op.getResult())
                        .template cast<base2::IEEE754Type>();
        auto dstType = converter->convertType(type);

        unsigned bitWidth = type.getBitWidth();
        if (bitWidth > 64)
            return rewriter.notifyMatchFailure(
                op,
                "expected at most 64-bit number");

        auto roundingMode = toLibRounding(adaptor.getRoundingMode());
        if (!roundingMode)
            return rewriter.notifyMatchFailure(
                op,
                "expected none or nearest rounding mode");

        Type i1Ty = IntegerType::get(rewriter.getContext(), 1);
        Type i8Ty = IntegerType::get(rewriter.getContext(), 8);
        Type i32Ty = IntegerType::get(rewriter.getContext(), 32);
        auto lhs = adaptor.getLhs();
        auto rhs = adaptor.getRhs();

        auto expBits = rewriter.create<arith::ConstantOp>(
            loc,
            IntegerAttr::get(i8Ty, type.getExponentBits()));
        auto fracBits = rewriter.create<arith::ConstantOp>(
            loc,
            IntegerAttr::get(i8Ty, type.getPrecision()));
        auto expBias = rewriter.create<arith::ConstantOp>(
            loc,
            IntegerAttr::get(i32Ty, -type.getBias()));
        auto hasRounding = rewriter.create<arith::ConstantOp>(
            loc,
            IntegerAttr::get(i1Ty, *roundingMode));
        auto hasNan =
            rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(i1Ty, 1));
        auto hasOne =
            rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(i1Ty, 1));
        auto hasSubnorm =
            rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(i1Ty, 1));
        auto sign =
            rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(i8Ty, -1));

        if (function == "add")
            rewriter.replaceOpWithNewOp<softfloat::AddOp>(
                op,
                dstType,
                lhs,
                rhs,
                expBits,
                fracBits,
                expBias,
                hasRounding,
                hasNan,
                hasOne,
                hasSubnorm,
                sign);
        if (function == "sub")
            rewriter.replaceOpWithNewOp<softfloat::SubOp>(
                op,
                dstType,
                lhs,
                rhs,
                expBits,
                fracBits,
                expBias,
                hasRounding,
                hasNan,
                hasOne,
                hasSubnorm,
                sign);
        if (function == "mul")
            rewriter.replaceOpWithNewOp<softfloat::MulOp>(
                op,
                dstType,
                lhs,
                rhs,
                expBits,
                fracBits,
                expBias,
                hasRounding,
                hasNan,
                hasOne,
                hasSubnorm,
                sign);
        if (function == "div")
            rewriter.replaceOpWithNewOp<softfloat::DivGOp>(
                op,
                dstType,
                lhs,
                rhs,
                expBits,
                fracBits,
                expBias,
                hasRounding,
                hasNan,
                hasOne,
                hasSubnorm,
                sign);

        return success();
    }

private:
    std::string function;
};

// Replace base2.cmp op with softfloat op
struct CompareOpLowering final : public OpConversionPattern<base2::CmpOp> {
public:
    using OpConversionPattern<base2::CmpOp>::OpConversionPattern;

    CompareOpLowering(
        TypeConverter &typeConverter,
        MLIRContext* context,
        PatternBenefit benefit)
            : OpConversionPattern<base2::CmpOp>(
                typeConverter,
                context,
                benefit){};

    LogicalResult matchAndRewrite(
        base2::CmpOp op,
        base2::CmpOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        assert(adaptor.getOperands().size() == 2);

        if (op.getResult().getType().template isa<ShapedType>())
            return rewriter.notifyMatchFailure(op, "expected scalar operation");

        Location loc = op->getLoc();

        TypeConverter* converter = this->typeConverter;
        auto type =
            getElementTypeOrSelf(op.getLhs())
                .template cast<base2::IEEE754Type>(); // don't need
                                                      // getElementTypeorSelf
        auto reslType = op.getResult().getType();     // same as above
        auto dstType = converter->convertType(reslType);

        unsigned bitWidth = type.getBitWidth();
        if (bitWidth > 64)
            return rewriter.notifyMatchFailure(
                op,
                "expected at most 64-bit number");

        auto pred = adaptor.getPredicate();
        auto predBit = toNanBit(pred);
        if (!predBit)
            return rewriter.notifyMatchFailure(op, "expected predicate");

        Type i1Ty = IntegerType::get(rewriter.getContext(), 1);
        Type i8Ty = IntegerType::get(rewriter.getContext(), 8);
        Type i32Ty = IntegerType::get(rewriter.getContext(), 32);
        auto lhs = adaptor.getLhs();
        auto rhs = adaptor.getRhs();

        auto expBits = rewriter.create<arith::ConstantOp>(
            loc,
            IntegerAttr::get(i8Ty, type.getExponentBits()));
        auto fracBits = rewriter.create<arith::ConstantOp>(
            loc,
            IntegerAttr::get(i8Ty, type.getPrecision()));
        auto expBias = rewriter.create<arith::ConstantOp>(
            loc,
            IntegerAttr::get(i32Ty, -type.getBias()));
        auto hasRounding =
            rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(i1Ty, 0));
        auto hasNan = rewriter.create<arith::ConstantOp>(
            loc,
            IntegerAttr::get(i1Ty, *predBit));
        auto hasOne =
            rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(i1Ty, 1));
        auto hasSubnorm =
            rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(i1Ty, 1));
        auto sign =
            rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(i8Ty, -1));

        if (pred == PartialOrderingPredicate::OrderedAndEqual
            || pred == PartialOrderingPredicate::UnorderedOrEqual)
            rewriter.replaceOpWithNewOp<softfloat::EQOp>(
                op,
                dstType,
                lhs,
                rhs,
                expBits,
                fracBits,
                expBias,
                hasRounding,
                hasNan,
                hasOne,
                hasSubnorm,
                sign);
        if (pred == PartialOrderingPredicate::OrderedAndGreater
            || pred == PartialOrderingPredicate::UnorderedOrGreater)
            rewriter.replaceOpWithNewOp<softfloat::GTOp>(
                op,
                dstType,
                lhs,
                rhs,
                expBits,
                fracBits,
                expBias,
                hasRounding,
                hasNan,
                hasOne,
                hasSubnorm,
                sign);
        if (pred == PartialOrderingPredicate::OrderedAndGreaterOrEqual
            || pred == PartialOrderingPredicate::UnorderedOrGreaterOrEqual)
            rewriter.replaceOpWithNewOp<softfloat::GEOp>(
                op,
                dstType,
                lhs,
                rhs,
                expBits,
                fracBits,
                expBias,
                hasRounding,
                hasNan,
                hasOne,
                hasSubnorm,
                sign);
        if (pred == PartialOrderingPredicate::OrderedAndLess
            || pred == PartialOrderingPredicate::UnorderedOrLess)
            rewriter.replaceOpWithNewOp<softfloat::LTOp>(
                op,
                dstType,
                lhs,
                rhs,
                expBits,
                fracBits,
                expBias,
                hasRounding,
                hasNan,
                hasOne,
                hasSubnorm,
                sign);
        if (pred == PartialOrderingPredicate::OrderedAndLessOrEqual
            || pred == PartialOrderingPredicate::UnorderedOrLessOrEqual)
            rewriter.replaceOpWithNewOp<softfloat::LEOp>(
                op,
                dstType,
                lhs,
                rhs,
                expBits,
                fracBits,
                expBias,
                hasRounding,
                hasNan,
                hasOne,
                hasSubnorm,
                sign);
        if (pred == PartialOrderingPredicate::OrderedAndUnequal
            || pred == PartialOrderingPredicate::UnorderedOrUnequal)
            rewriter.replaceOpWithNewOp<softfloat::LTGTOp>(
                op,
                dstType,
                lhs,
                rhs,
                expBits,
                fracBits,
                expBias,
                hasRounding,
                hasNan,
                hasOne,
                hasSubnorm,
                sign);

        return success();
    }
};

// Replace base2.bit_cast op with softfloat op
struct ValueCastOpLowering final
        : public OpConversionPattern<base2::ValueCastOp> {
public:
    using OpConversionPattern<base2::ValueCastOp>::OpConversionPattern;

    ValueCastOpLowering(
        TypeConverter &typeConverter,
        MLIRContext* context,
        PatternBenefit benefit)
            : OpConversionPattern<base2::ValueCastOp>(
                typeConverter,
                context,
                benefit){};

    LogicalResult matchAndRewrite(
        base2::ValueCastOp op,
        base2::ValueCastOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        assert(adaptor.getOperands().size() == 1);

        if (op.getResult().getType().template isa<ShapedType>())
            return rewriter.notifyMatchFailure(op, "expected scalar operation");

        Location loc = op->getLoc();

        TypeConverter* converter = this->typeConverter;

        auto inTy = op.getIn().getType();
        auto outTy = op.getOut().getType();
        Type dstType;
        ValueRange args;
        Type i1Ty = IntegerType::get(rewriter.getContext(), 1);
        Type i8Ty = IntegerType::get(rewriter.getContext(), 8);
        Type i32Ty = IntegerType::get(rewriter.getContext(), 32);

        auto in = adaptor.getIn();
        auto hasRounding =
            rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(i1Ty, 1));
        auto hasNan =
            rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(i1Ty, 1));
        auto hasOne =
            rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(i1Ty, 1));
        auto hasSubnorm =
            rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(i1Ty, 1));
        auto sign =
            rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(i8Ty, -1));

        if (inTy.dyn_cast<FloatType>() && outTy.dyn_cast<IEEE754Semantics>()) {
            auto outRealTy = outTy.template cast<IEEE754Type>();
            if (outRealTy.getBitWidth() > 64)
                return rewriter.notifyMatchFailure(
                    op,
                    "expected at most 64-bit number");
            dstType = converter->convertType(outRealTy);

            auto expBits = rewriter.create<arith::ConstantOp>(
                loc,
                IntegerAttr::get(i8Ty, outRealTy.getExponentBits()));
            auto fracBits = rewriter.create<arith::ConstantOp>(
                loc,
                IntegerAttr::get(i8Ty, outRealTy.getPrecision()));
            auto expBias = rewriter.create<arith::ConstantOp>(
                loc,
                IntegerAttr::get(i32Ty, -outRealTy.getBias()));
            rewriter.replaceOpWithNewOp<softfloat::CastFloatOp>(
                op,
                dstType,
                in,
                expBits,
                fracBits,
                expBias,
                hasRounding,
                hasNan,
                hasOne,
                hasSubnorm,
                sign);
        }

        if (inTy.dyn_cast<IEEE754Semantics>() && outTy.dyn_cast<FloatType>()) {
            auto inRealTy = inTy.template cast<IEEE754Type>();
            if (inRealTy.getBitWidth() > 64)
                return rewriter.notifyMatchFailure(
                    op,
                    "expected at most 64-bit number");
            dstType = converter->convertType(outTy);

            auto expBits = rewriter.create<arith::ConstantOp>(
                loc,
                IntegerAttr::get(i8Ty, inRealTy.getExponentBits()));
            auto fracBits = rewriter.create<arith::ConstantOp>(
                loc,
                IntegerAttr::get(i8Ty, inRealTy.getPrecision()));
            auto expBias = rewriter.create<arith::ConstantOp>(
                loc,
                IntegerAttr::get(i32Ty, -inRealTy.getBias()));
            rewriter.replaceOpWithNewOp<softfloat::CastToFloatOp>(
                op,
                dstType,
                in,
                expBits,
                fracBits,
                expBias,
                hasRounding,
                hasNan,
                hasOne,
                hasSubnorm,
                sign);
        }

        if (inTy.dyn_cast<IEEE754Semantics>()
            && outTy.dyn_cast<IEEE754Semantics>()) {
            auto inRealTy = inTy.template cast<IEEE754Type>();
            auto outRealTy = outTy.template cast<IEEE754Type>();
            if (inRealTy.getBitWidth() > 64)
                return rewriter.notifyMatchFailure(
                    op,
                    "expected at most 64-bit number");
            if (outRealTy.getBitWidth() > 64)
                return rewriter.notifyMatchFailure(
                    op,
                    "expected at most 64-bit number");
            dstType = converter->convertType(outRealTy);

            auto in_expBits = rewriter.create<arith::ConstantOp>(
                loc,
                IntegerAttr::get(i8Ty, inRealTy.getExponentBits()));
            auto in_fracBits = rewriter.create<arith::ConstantOp>(
                loc,
                IntegerAttr::get(i8Ty, inRealTy.getPrecision()));
            auto in_expBias = rewriter.create<arith::ConstantOp>(
                loc,
                IntegerAttr::get(i32Ty, -inRealTy.getBias()));
            auto out_expBits = rewriter.create<arith::ConstantOp>(
                loc,
                IntegerAttr::get(i8Ty, outRealTy.getExponentBits()));
            auto out_fracBits = rewriter.create<arith::ConstantOp>(
                loc,
                IntegerAttr::get(i8Ty, outRealTy.getPrecision()));
            auto out_expBias = rewriter.create<arith::ConstantOp>(
                loc,
                IntegerAttr::get(i32Ty, -outRealTy.getBias()));
            rewriter.replaceOpWithNewOp<softfloat::CastOp>(
                op,
                dstType,
                in,
                in_expBits,
                in_fracBits,
                in_expBias,
                hasRounding,
                hasNan,
                hasOne,
                hasSubnorm,
                sign,
                out_expBits,
                out_fracBits,
                out_expBias,
                hasRounding,
                hasNan,
                hasOne,
                hasSubnorm,
                sign);
        }

        return success();
    }
};

// Replace base2.max/min op with softfloat op
// template<typename Op>
// struct MinMaxOpLowering final : public OpConversionPattern<Op> {
// public:
//     using OpConversionPattern<Op>::OpConversionPattern;

//     MinMaxOpLowering<Op>(
//         TypeConverter &typeConverter,
//         MLIRContext* context,
//         StringRef function,
//         PatternBenefit benefit)
//             : OpConversionPattern<Op>(typeConverter, context, benefit),
//               function(function){};

//     LogicalResult matchAndRewrite(
//         Op op,
//         typename Op::Adaptor adaptor,
//         ConversionPatternRewriter &rewriter) const override
//     {
//         assert(adaptor.getOperands().size() == 2);

//         if (op.getResult().getType().template isa<ShapedType>())
//             return rewriter.notifyMatchFailure(op, "expected scalar
//             operation");

//         Location loc = op->getLoc();

//         TypeConverter* converter = this->typeConverter;
//         auto type = getElementTypeOrSelf(op.getResult())
//                         .template cast<base2::IEEE754Type>();
//         auto dstType = converter->convertType(type);

//         unsigned outBitWidth = type.getBitWidth();
//         if (outBitWidth > 64)
//             return rewriter.notifyMatchFailure(
//                 op,
//                 "expected at most 64-bit number");

//         auto lhs = adaptor.getLhs();
//         auto rhs = adaptor.getRhs();

//         Value cmp_resl;
//         if (function == "max")
//             cmp_resl = rewriter.create<base2::CmpOp>(
//                 loc,
//                 IntegerType::get(rewriter.getContext(), 1),
//                 PartialOrderingPredicate::OrderedAndGreater,
//                 op.getLhs(),
//                 op.getRhs());
//         if (function == "min")
//             cmp_resl = rewriter.create<base2::CmpOp>(
//                 loc,
//                 IntegerType::get(rewriter.getContext(), 1),
//                 PartialOrderingPredicate::UnorderedOrLess,
//                 op.getLhs(),
//                 op.getRhs()); // NaN as missing number as IEEE754 old, think
//                 of
//                               // a better way?

//         rewriter.replaceOpWithNewOp<arith::SelectOp>(
//             op,
//             dstType,
//             cmp_resl,
//             lhs,
//             rhs);

//         return success();
//     }

// private:
//     std::string function;
// };

} // namespace

void mlir::populateBase2ToSoftFloatConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns,
    PatternBenefit benefit)
{
    patterns.add<ClosedArithOpLowering<base2::AddOp>>(
        typeConverter,
        patterns.getContext(),
        "add",
        benefit);
    patterns.add<ClosedArithOpLowering<base2::SubOp>>(
        typeConverter,
        patterns.getContext(),
        "sub",
        benefit);
    patterns.add<ClosedArithOpLowering<base2::MulOp>>(
        typeConverter,
        patterns.getContext(),
        "mul",
        benefit);
    patterns.add<ClosedArithOpLowering<base2::DivOp>>(
        typeConverter,
        patterns.getContext(),
        "div",
        benefit);
    patterns.add<CompareOpLowering>(
        typeConverter,
        patterns.getContext(),
        benefit);
    patterns.add<ValueCastOpLowering>(
        typeConverter,
        patterns.getContext(),
        benefit);
    // patterns.add<MinMaxOpLowering<base2::MinOp>>(
    //     typeConverter,
    //     patterns.getContext(),
    //     "min",
    //     benefit);
    // patterns.add<MinMaxOpLowering<base2::MaxOp>>(
    //     typeConverter,
    //     patterns.getContext(),
    //     "max",
    //     benefit);
}

namespace {
struct ConvertBase2ToSoftFloatPass
        : public ConvertBase2ToSoftFloatBase<ConvertBase2ToSoftFloatPass> {
    void runOnOperation() final;
};
} // namespace

/***
 * Conversion Target
 ***/

void ConvertBase2ToSoftFloatPass::runOnOperation()
{
    TypeConverter converter;

    // Allow all unknown types.
    converter.addConversion([&](Type type) { return type; });

    // Convert IEEE754 Type to SFloatType type.
    converter.addConversion([&](IEEE754Type type) -> Type {
        return converter.convertType(
            softfloat::SFloatType::get(type.getContext()));
    });
    // For any shaped type, do the same thing as above
    converter.addConversion([&](ShapedType shapedTy) -> Type {
        return shapedTy.cloneWith(
            std::nullopt,
            converter.convertType(shapedTy.getElementType()));
    });

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    const auto addUnrealizedCast =
        [](OpBuilder &builder, Type type, ValueRange inputs, Location loc) {
            return builder.create<UnrealizedConversionCastOp>(loc, type, inputs)
                .getResult(0);
        };
    converter.addSourceMaterialization(addUnrealizedCast);
    converter.addTargetMaterialization(addUnrealizedCast);
    target.addLegalOp<UnrealizedConversionCastOp>();

    // Convert elementwise operations to linalg
    linalg::populateElementwiseToLinalgConversionPatterns(patterns);

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
    populateBase2ToSoftFloatConversionPatterns(converter, patterns, 1);

    // Remove unrealized casts wherever possible.
    populateReconcileUnrealizedCastsPatterns(patterns);

    target.addDynamicallyLegalOp<base2::ValueCastOp>(
        [&](Operation* op) { return converter.isLegal(op); });
    target.addLegalDialect<
        BuiltinDialect,
        arith::ArithDialect,
        linalg::LinalgDialect,
        memref::MemRefDialect,
        softfloat::SoftFloatDialect,
        tensor::TensorDialect>();
    target.addIllegalDialect<base2::Base2Dialect>();

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns))))
        signalPassFailure();
}

std::unique_ptr<Pass> mlir::createConvertBase2ToSoftFloatPass()
{
    return std::make_unique<ConvertBase2ToSoftFloatPass>();
}
