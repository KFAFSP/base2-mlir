/// Implements the SoftFloatToLib pass.
///
/// @file
/// @author     Jihaong Bi (jiahong.bi@mailbox.tu-dresden.de)

#include "base2-mlir/Conversion/SoftFloatToLib/SoftFloatToLib.h"

#include "../PassDetails.h"
#include "base2-mlir/Dialect/SoftFloat/IR/SoftFloat.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace mlir::softfloat;

namespace {

// Pattern to convert operations to calls to softfloat lib functions
template<typename Op>
struct SoftFloatOpToLibCall final : public OpConversionPattern<Op> {
public:
    using OpConversionPattern<Op>::OpConversionPattern;

    SoftFloatOpToLibCall<Op>(
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
        // softfloat lib functions take maximal 17 operands
        assert(
            adaptor.getOperands().size() == 9
            || adaptor.getOperands().size() == 10
            || adaptor.getOperands().size() == 17);

        Operation* module = op->template getParentOfType<ModuleOp>();

        TypeConverter* converter = this->typeConverter;
        SymbolTableCollection symbolTable; // new SymbolTable for this operation

        auto type = op.getResult().getType();
        auto dstType = converter->convertType(type);
        if (!type.template isa<softfloat::SFloatType, IntegerType>())
            return failure();

        auto name = function;
        auto funcAttr = StringAttr::get(op->getContext(), name);
        auto opFunc = symbolTable.lookupNearestSymbolFrom<SymbolOpInterface>(
            op,
            funcAttr);

        if (!opFunc) {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(&module->getRegion(0).front());
            auto opFunctionType = FunctionType::get(
                rewriter.getContext(),
                adaptor.getOperands().getTypes(),
                dstType);
            opFunc = rewriter.create<func::FuncOp>(
                rewriter.getUnknownLoc(),
                name,
                opFunctionType);
            opFunc.setPrivate();
        }
        assert(isa<FunctionOpInterface>(
            SymbolTable::lookupSymbolIn(module, name)));

        rewriter.replaceOpWithNewOp<func::CallOp>(
            op,
            name,
            dstType,
            adaptor.getOperands());

        return success();
    }

private:
    std::string function;
};

// Lowering the softfloat.castfloat operation
struct SoftFloatCastFloatLowering final
        : public OpConversionPattern<softfloat::CastFloatOp> {
public:
    using OpConversionPattern<softfloat::CastFloatOp>::OpConversionPattern;

    SoftFloatCastFloatLowering(
        TypeConverter &typeConverter,
        MLIRContext* context,
        PatternBenefit benefit)
            : OpConversionPattern<softfloat::CastFloatOp>(
                typeConverter,
                context,
                benefit){};

    LogicalResult matchAndRewrite(
        softfloat::CastFloatOp op,
        softfloat::CastFloatOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // softfloat lib functions take exactly 9 operands
        assert(adaptor.getOperands().size() == 9);

        Location loc = op->getLoc();

        TypeConverter* converter = this->typeConverter;
        SymbolTableCollection symbolTable; // new SymbolTable for this operation

        auto type = op.getResult().getType();
        auto dstType = converter->convertType(type);
        if (!type.template isa<softfloat::SFloatType>()) return failure();

        Type i1Ty = IntegerType::get(rewriter.getContext(), 1);
        Type i8Ty = IntegerType::get(rewriter.getContext(), 8);
        Type i32Ty = IntegerType::get(rewriter.getContext(), 32);
        Type i64Ty = IntegerType::get(rewriter.getContext(), 64);
        auto in = adaptor.getIn();
        auto in_bitcast = rewriter.create<arith::BitcastOp>(loc, i64Ty, in);
        auto in_expBits =
            rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(i8Ty, 11));
        auto in_fracBits =
            rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(i8Ty, 52));
        auto in_expBias = rewriter.create<arith::ConstantOp>(
            loc,
            IntegerAttr::get(i32Ty, -1023));
        auto in_hasRounding =
            rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(i1Ty, 1));
        auto in_hasNan =
            rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(i1Ty, 1));
        auto in_hasOne =
            rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(i1Ty, 1));
        auto in_hasSubnorm =
            rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(i1Ty, 1));
        auto in_sign = rewriter.create<arith::ConstantOp>(
            loc,
            IntegerAttr::get(i8Ty, -1)); // automatical sign

        auto out_expBits = adaptor.getExpBits();
        auto out_fracBits = adaptor.getFracBits();
        auto out_expBias = adaptor.getExpBias();
        auto out_hasRounding = adaptor.getHasRounding();
        auto out_hasNan = adaptor.getHasNan();
        auto out_hasOne = adaptor.getHasOne();
        auto out_hasSubnorm = adaptor.getHasSubnorm();
        auto out_sign = adaptor.getSign();

        rewriter.replaceOpWithNewOp<softfloat::CastOp>(
            op,
            dstType,
            in_bitcast,
            in_expBits,
            in_fracBits,
            in_expBias,
            in_hasRounding,
            in_hasNan,
            in_hasOne,
            in_hasSubnorm,
            in_sign,
            out_expBits,
            out_fracBits,
            out_expBias,
            out_hasRounding,
            out_hasNan,
            out_hasOne,
            out_hasSubnorm,
            out_sign);

        return success();
    }

private:
    std::string function;
};

// Lowering the softfloat.casttofloat operation
struct SoftFloatCastToFloatLowering final
        : public OpConversionPattern<softfloat::CastToFloatOp> {
public:
    using OpConversionPattern<softfloat::CastToFloatOp>::OpConversionPattern;

    SoftFloatCastToFloatLowering(
        TypeConverter &typeConverter,
        MLIRContext* context,
        PatternBenefit benefit)
            : OpConversionPattern<softfloat::CastToFloatOp>(
                typeConverter,
                context,
                benefit){};

    LogicalResult matchAndRewrite(
        softfloat::CastToFloatOp op,
        softfloat::CastToFloatOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // softfloat lib functions take exactly 9 operands
        assert(adaptor.getOperands().size() == 9);

        Location loc = op->getLoc();

        TypeConverter* converter = this->typeConverter;
        SymbolTableCollection symbolTable; // new SymbolTable for this operation

        auto type = op.getResult().getType();
        auto dstType = converter->convertType(type);
        if (!type.template isa<FloatType>()) return failure();

        Type i1Ty = IntegerType::get(rewriter.getContext(), 1);
        Type i8Ty = IntegerType::get(rewriter.getContext(), 8);
        Type i32Ty = IntegerType::get(rewriter.getContext(), 32);
        auto in = adaptor.getIn();
        auto in_expBits = adaptor.getExpBits();
        auto in_fracBits = adaptor.getFracBits();
        auto in_expBias = adaptor.getExpBias();
        auto in_hasRounding = adaptor.getHasRounding();
        auto in_hasNan = adaptor.getHasNan();
        auto in_hasOne = adaptor.getHasOne();
        auto in_hasSubnorm = adaptor.getHasSubnorm();
        auto in_sign = adaptor.getSign();

        auto out_expBits =
            rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(i8Ty, 11));
        auto out_fracBits =
            rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(i8Ty, 52));
        auto out_expBias = rewriter.create<arith::ConstantOp>(
            loc,
            IntegerAttr::get(i32Ty, -1023));
        auto out_hasRounding =
            rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(i1Ty, 1));
        auto out_hasNan =
            rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(i1Ty, 1));
        auto out_hasOne =
            rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(i1Ty, 1));
        auto out_hasSubnorm =
            rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(i1Ty, 1));
        auto out_sign =
            rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(i8Ty, -1));

        auto out = rewriter.create<softfloat::CastOp>(
            loc,
            in,
            in_expBits,
            in_fracBits,
            in_expBias,
            in_hasRounding,
            in_hasNan,
            in_hasOne,
            in_hasSubnorm,
            in_sign,
            out_expBits,
            out_fracBits,
            out_expBias,
            out_hasRounding,
            out_hasNan,
            out_hasOne,
            out_hasSubnorm,
            out_sign);

        auto out_value = rewriter.getRemappedValue(out);

        rewriter.replaceOpWithNewOp<arith::BitcastOp>(op, dstType, out_value);

        return success();
    }

private:
    std::string function;
};

// Rewrite affine.load/store to cast sfloat type into i64 within memref
struct AffineLoadRewriting final : public OpConversionPattern<AffineLoadOp> {
public:
    using OpConversionPattern<AffineLoadOp>::OpConversionPattern;

    AffineLoadRewriting(
        TypeConverter &typeConverter,
        MLIRContext* context,
        PatternBenefit benefit)
            : OpConversionPattern<AffineLoadOp>(
                typeConverter,
                context,
                benefit){};

    LogicalResult matchAndRewrite(
        AffineLoadOp op,
        AffineLoadOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto type = op.getResult().getType();
        if (!type.template isa<softfloat::SFloatType>()) return failure();

        rewriter.replaceOpWithNewOp<AffineLoadOp>(
            op,
            adaptor.getMemref(),
            op.getAffineMap(),
            op.getMapOperands());

        // affine -> memref
        // auto loc = op.getLoc();
        // auto map = op.getAffineMap();

        // auto resultOperands =
        //     expandAffineMap(rewriter, loc, map, adaptor.getIndices());
        // if (!resultOperands) return failure();

        // rewriter.replaceOpWithNewOp<memref::LoadOp>(
        //     op,
        //     adaptor.getMemref(),
        //     *resultOperands);

        return success();
    }
};

struct AffineStoreRewriting final : public OpConversionPattern<AffineStoreOp> {
public:
    using OpConversionPattern<AffineStoreOp>::OpConversionPattern;

    AffineStoreRewriting(
        TypeConverter &typeConverter,
        MLIRContext* context,
        PatternBenefit benefit)
            : OpConversionPattern<AffineStoreOp>(
                typeConverter,
                context,
                benefit){};

    LogicalResult matchAndRewrite(
        AffineStoreOp op,
        AffineStoreOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto type = op.getValue().getType();
        if (!type.template isa<softfloat::SFloatType>()) return failure();

        rewriter.replaceOpWithNewOp<AffineStoreOp>(
            op,
            adaptor.getValue(),
            adaptor.getMemref(),
            op.getAffineMap(),
            op.getMapOperands());

        // affine -> memref
        // auto loc = op.getLoc();
        // auto map = op.getAffineMap();

        // auto resultOperands =
        //     expandAffineMap(rewriter, loc, map, adaptor.getIndices());
        // if (!resultOperands) return failure();

        // rewriter.replaceOpWithNewOp<memref::StoreOp>(
        //     op,
        //     adaptor.getValue(),
        //     adaptor.getMemref(),
        //     *resultOperands);

        return success();
    }
};

// Rewrite memref.alloc
struct MemRefAllocRewriting final
        : public OpConversionPattern<memref::AllocOp> {
public:
    using OpConversionPattern<memref::AllocOp>::OpConversionPattern;

    MemRefAllocRewriting(
        TypeConverter &typeConverter,
        MLIRContext* context,
        PatternBenefit benefit)
            : OpConversionPattern<memref::AllocOp>(
                typeConverter,
                context,
                benefit){};

    LogicalResult matchAndRewrite(
        memref::AllocOp op,
        memref::AllocOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        Type newMemRefTy = getTypeConverter()->convertType(op.getType());

        rewriter.replaceOpWithNewOp<memref::AllocOp>(
            op,
            newMemRefTy,
            adaptor.getDynamicSizes(),
            adaptor.getSymbolOperands(),
            adaptor.getAlignmentAttr());

        return success();
    }
};

// Rewrite memref.load/store to cast sfloat type into i64 within memref
struct MemRefLoadRewriting final : public OpConversionPattern<memref::LoadOp> {
public:
    using OpConversionPattern<memref::LoadOp>::OpConversionPattern;

    MemRefLoadRewriting(
        TypeConverter &typeConverter,
        MLIRContext* context,
        PatternBenefit benefit)
            : OpConversionPattern<memref::LoadOp>(
                typeConverter,
                context,
                benefit){};

    LogicalResult matchAndRewrite(
        memref::LoadOp op,
        memref::LoadOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto type = op.getResult().getType();
        if (!type.template isa<softfloat::SFloatType>()) return failure();

        rewriter.replaceOpWithNewOp<memref::LoadOp>(
            op,
            adaptor.getMemref(),
            adaptor.getIndices());

        return success();
    }
};

struct MemRefStoreRewriting final
        : public OpConversionPattern<memref::StoreOp> {
public:
    using OpConversionPattern<memref::StoreOp>::OpConversionPattern;

    MemRefStoreRewriting(
        TypeConverter &typeConverter,
        MLIRContext* context,
        PatternBenefit benefit)
            : OpConversionPattern<memref::StoreOp>(
                typeConverter,
                context,
                benefit){};

    LogicalResult matchAndRewrite(
        memref::StoreOp op,
        memref::StoreOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto type = op.getValue().getType();
        if (!type.template isa<softfloat::SFloatType>()) return failure();

        rewriter.replaceOpWithNewOp<memref::StoreOp>(
            op,
            adaptor.getValue(),
            adaptor.getMemref(),
            adaptor.getIndices());

        return success();
    }
};

// Rewrite linalg.generic
/*struct LinalgGenericRewriting final
        : public OpConversionPattern<linalg::GenericOp> {
public:
    using OpConversionPattern<linalg::GenericOp>::OpConversionPattern;

    LinalgGenericRewriting(
        TypeConverter &typeConverter,
        MLIRContext* context,
        PatternBenefit benefit)
            : OpConversionPattern<linalg::GenericOp>(
                typeConverter,
                context,
                benefit){};

    LogicalResult matchAndRewrite(
        linalg::GenericOp op,
        linalg::GenericOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto type = op.getResultTypes()[0];
        if (!type.template isa<softfloat::SFloatType>()) return failure();

        TypeConverter* converter = this->typeConverter;
        auto results = op.getResults();
        SmallVector<Type> opResultTypes;

        for (auto result : results) {
            auto resultTy = converter->convertType(result.getType());
            opResultTypes.push_back(resultTy);
        }

        // auto i8Ty = IntegerType::get(rewriter.getContext(), 8);

        rewriter.replaceOpWithNewOp<linalg::GenericOp>(
            op,
            opResultTypes,
            adaptor.getInputs(),
            adaptor.getOutputs(),
            adaptor.getIndexingMapsAttr(),
            adaptor.getIteratorTypesAttr(),
            adaptor.getDocAttr(),
            adaptor.getLibraryCallAttr(),
            [&](OpBuilder &builder, Location loc, ValueRange args) {
            }); // TODO: rewrite this op

        return success();
    }
}; */

// Rewrite linalg.yield
/* struct LinalgYieldRewriting final
        : public OpConversionPattern<linalg::YieldOp> {
public:
    using OpConversionPattern<linalg::YieldOp>::OpConversionPattern;

    LinalgYieldRewriting(
        TypeConverter &typeConverter,
        MLIRContext* context,
        PatternBenefit benefit)
            : OpConversionPattern<linalg::YieldOp>(
                typeConverter,
                context,
                benefit){};

    LogicalResult matchAndRewrite(
        linalg::YieldOp op,
        linalg::YieldOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // TypeConverter* converter = this->typeConverter;
        auto type = op.getValues().getTypes().front();
        if (!type.template isa<softfloat::SFloatType>()) return failure();

        rewriter.replaceOpWithNewOp<linalg::YieldOp>(op, adaptor.getValues());

        return success();
    }
}; */

} // namespace

void mlir::populateSoftFloatToLibConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns,
    PatternBenefit benefit)
{
    patterns.add<SoftFloatOpToLibCall<softfloat::AddOp>>(
        typeConverter,
        patterns.getContext(),
        "__float_add",
        benefit);
    patterns.add<SoftFloatOpToLibCall<softfloat::SubOp>>(
        typeConverter,
        patterns.getContext(),
        "__float_sub",
        benefit);
    patterns.add<SoftFloatOpToLibCall<softfloat::MulOp>>(
        typeConverter,
        patterns.getContext(),
        "__float_mul",
        benefit);
    patterns.add<SoftFloatOpToLibCall<softfloat::DivSRTOp>>(
        typeConverter,
        patterns.getContext(),
        "__float_divSRT4",
        benefit);
    patterns.add<SoftFloatOpToLibCall<softfloat::DivGOp>>(
        typeConverter,
        patterns.getContext(),
        "__float_divG",
        benefit);
    patterns.add<SoftFloatOpToLibCall<softfloat::EQOp>>(
        typeConverter,
        patterns.getContext(),
        "__float_eq",
        benefit);
    patterns.add<SoftFloatOpToLibCall<softfloat::LEOp>>(
        typeConverter,
        patterns.getContext(),
        "__float_le",
        benefit);
    patterns.add<SoftFloatOpToLibCall<softfloat::LTOp>>(
        typeConverter,
        patterns.getContext(),
        "__float_lt",
        benefit);
    patterns.add<SoftFloatOpToLibCall<softfloat::GEOp>>(
        typeConverter,
        patterns.getContext(),
        "__float_ge",
        benefit);
    patterns.add<SoftFloatOpToLibCall<softfloat::GTOp>>(
        typeConverter,
        patterns.getContext(),
        "__float_gt",
        benefit);
    patterns.add<SoftFloatOpToLibCall<softfloat::LTGTOp>>(
        typeConverter,
        patterns.getContext(),
        "__float_ltgt_quiet",
        benefit);
    patterns.add<SoftFloatOpToLibCall<softfloat::NaNOp>>(
        typeConverter,
        patterns.getContext(),
        "__float64_is_signaling_nan",
        benefit);
    patterns.add<SoftFloatOpToLibCall<softfloat::CastOp>>(
        typeConverter,
        patterns.getContext(),
        "__float_cast",
        benefit);
    patterns.add<SoftFloatCastFloatLowering>(
        typeConverter,
        patterns.getContext(),
        benefit);
    patterns.add<SoftFloatCastToFloatLowering>(
        typeConverter,
        patterns.getContext(),
        benefit);
    patterns.add<AffineLoadRewriting>(
        typeConverter,
        patterns.getContext(),
        benefit);
    patterns.add<AffineStoreRewriting>(
        typeConverter,
        patterns.getContext(),
        benefit);
    patterns.add<MemRefLoadRewriting>(
        typeConverter,
        patterns.getContext(),
        benefit);
    patterns.add<MemRefStoreRewriting>(
        typeConverter,
        patterns.getContext(),
        benefit);
    patterns.add<MemRefAllocRewriting>(
        typeConverter,
        patterns.getContext(),
        benefit);
    /* patterns.add<LinalgGenericRewriting>(
        typeConverter,
        patterns.getContext(),
        benefit);
    patterns.add<LinalgYieldRewriting>(
        typeConverter,
        patterns.getContext(),
        benefit); */
}

namespace {
struct ConvertSoftFloatToLibPass
        : public ConvertSoftFloatToLibBase<ConvertSoftFloatToLibPass> {
    void runOnOperation() final;
};
} // namespace

/***
 * Conversion Target
 ***/

void ConvertSoftFloatToLibPass::runOnOperation()
{
    TypeConverter converter;

    // Allow all unknown types.
    converter.addConversion([&](Type type) { return type; });

    // Convert SFloatType type to the underlying implementation (i64) type.
    converter.addConversion([&](softfloat::SFloatType type) -> Type {
        return converter.convertType(IntegerType::get(type.getContext(), 64));
    });
    // For any shaped type, do the same thing as above
    converter.addConversion([&](ShapedType shapedTy) -> Type {
        return shapedTy.cloneWith(
            std::nullopt,
            converter.convertType(shapedTy.getElementType()));
    });

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    // NOTE: Using SFloatType outside of the ops defined here is undefined
    //       behavior, which will remain present as unrealized casts.
    const auto addUnrealizedCast =
        [](OpBuilder &builder, Type type, ValueRange inputs, Location loc) {
            return builder.create<UnrealizedConversionCastOp>(loc, type, inputs)
                .getResult(0);
        };
    converter.addSourceMaterialization(addUnrealizedCast);
    converter.addTargetMaterialization(addUnrealizedCast);
    target.addLegalOp<UnrealizedConversionCastOp>();

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

    // Convert softfloat dialect operations.
    populateSoftFloatToLibConversionPatterns(converter, patterns, 1);

    // Remove unrealized casts wherever possible.
    populateReconcileUnrealizedCastsPatterns(patterns);

    target.addDynamicallyLegalOp<
        AffineLoadOp,
        AffineStoreOp,
        memref::AllocOp,
        memref::LoadOp,
        memref::StoreOp/*,
        linalg::GenericOp,
        linalg::YieldOp*/>([&](Operation* op) { return converter.isLegal(op); });
    target.addLegalDialect<
        BuiltinDialect,
        arith::ArithDialect,
        func::FuncDialect>();
    target.addIllegalDialect<softfloat::SoftFloatDialect>();

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns))))
        signalPassFailure();
}

std::unique_ptr<Pass> mlir::createConvertSoftFloatToLibPass()
{
    return std::make_unique<ConvertSoftFloatToLibPass>();
}
