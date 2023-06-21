/// Implements the ConvertBitToLLVMPass.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "base2-mlir/Conversion/BitToLLVM/BitToLLVM.h"
#include "base2-mlir/Dialect/Bit/IR/Bit.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::bit;

//===- Generated includes -------------------------------------------------===//

namespace mlir {

#define GEN_PASS_DEF_CONVERTBITTOLLVM
#include "base2-mlir/Conversion/Passes.h.inc"

} // namespace mlir

//===----------------------------------------------------------------------===//

namespace {

struct ConvertBitToLLVMPass
        : mlir::impl::ConvertBitToLLVMBase<ConvertBitToLLVMPass> {
    using ConvertBitToLLVMBase::ConvertBitToLLVMBase;

    void runOnOperation() override;
};

struct ConvertConstant : ConvertOpToLLVMPattern<ConstantOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(
        ConstantOp op,
        ConstantOp::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // Get the appropriate bit constant value.
        auto value = adaptor.getValue();
        if (!LLVM::isCompatibleType(value.getType())) {
            // Convert to machine integer bits.
            const auto bitWidth =
                value.getType().getElementType().getBitWidth();
            const auto machineTy =
                rewriter.getIntegerType(bitWidth).dyn_cast<BitSequenceType>();
            value = adaptor.getValue().bitCastElements(machineTy);
        }

        // Ensure we are using built-in attributes.
        value = BitSequenceLikeAttr::get(value);

        // Get the TyedAttr from BitSequenceAttr
        TypedAttr attr;
        if (auto intTy = dyn_cast<IntegerType>(value.getType())) {
            unsigned resultBitwidth = intTy.getWidth();
            auto attrTy = rewriter.getIntegerType(resultBitwidth);
            attr = rewriter.getIntegerAttr(attrTy, resultBitwidth);
        }

        // Create the LLVM mlir.constant operation.
        // NOTE: If we changed the type, the ConversionPatternRewriter will
        //       automatically insert a SourceMaterialization.
        rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(op, attr);
        return success();
    }
};

struct ConvertCast : OpRewritePattern<CastOp> {
    explicit ConvertCast(LLVMTypeConverter &typeConverter)
            : OpRewritePattern(&typeConverter.getContext()),
              typeConverter(&typeConverter)
    {}

    static Value materialize(
        OpBuilder &builder,
        LLVMTypeConverter &converter,
        Location loc,
        Type outTy,
        Value in)
    {
        if (!LLVM::isCompatibleType(in.getType())) {
            // Convert the input to a machine type via unspecified cast.
            const auto interTy = converter.convertType(in.getType());
            in = builder.create<UnrealizedConversionCastOp>(loc, interTy, in)
                     .getResult(0);
        }

        // Perform an LLVM bitcast.
        const auto interTy = LLVM::isCompatibleType(outTy)
                                 ? outTy
                                 : converter.convertType(outTy);
        in = builder.create<LLVM::BitcastOp>(loc, interTy, in);

        if (interTy != outTy) {
            // Convert the output from a machine type via unspecified cast.
            in = builder.create<UnrealizedConversionCastOp>(loc, outTy, in)
                     .getResult(0);
        }
        return in;
    }

    LogicalResult
    matchAndRewrite(CastOp op, PatternRewriter &rewriter) const override
    {
        rewriter.replaceOp(
            op,
            materialize(
                rewriter,
                *typeConverter,
                op.getLoc(),
                op.getType(),
                op.getIn()));
        return success();
    }

private:
    LLVMTypeConverter* typeConverter;
};

struct ConvertCmp : ConvertOpToLLVMPattern<CmpOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(
        CmpOp op,
        CmpOp::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // Convert the predicate.
        LLVM::ICmpPredicate pred;
        switch (op.getPredicate()) {
        case EqualityPredicate::Verum:
        case EqualityPredicate::Falsum:
            return rewriter.notifyMatchFailure(op, "incomplete folding");

        case EqualityPredicate::Equal: pred = LLVM::ICmpPredicate::eq; break;
        case EqualityPredicate::Unequal: pred = LLVM::ICmpPredicate::ne; break;
        }

        // Create the LLVM icmp operation.
        rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(
            op,
            pred,
            adaptor.getLhs(),
            adaptor.getRhs());
        return success();
    }
};

using ConvertSelect = OneToOneConvertToLLVMPattern<SelectOp, LLVM::SelectOp>;

using ConvertAnd = OneToOneConvertToLLVMPattern<AndOp, LLVM::AndOp>;
using ConvertOr = OneToOneConvertToLLVMPattern<OrOp, LLVM::OrOp>;
using ConvertXor = OneToOneConvertToLLVMPattern<XorOp, LLVM::XOrOp>;

template<class From, class To>
struct ConvertShiftOp : ConvertOpToLLVMPattern<From> {
    using ConvertOpToLLVMPattern<From>::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(
        From op,
        typename From::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        if (adaptor.getFunnel()) {
            // BUG: Add funnel shift intrinsic to LLVM dialect.
            return rewriter.notifyMatchFailure(
                op,
                "funnel shift not yet supported");
        }

        // Calculate the maximum permissible shift amount and clamp the amount.
        // This also ensures no lossy truncation occurs on index to machine
        // integer conversion.
        const auto bitWidth =
            adaptor.getValue().getType().getIntOrFloatBitWidth();
        const auto maxShift =
            rewriter.create<index::ConstantOp>(op.getLoc(), bitWidth)
                .getResult();
        const auto amount =
            rewriter
                .create<index::MinUOp>(op.getLoc(), op.getAmount(), maxShift)
                .getResult();
        // Convert the shift amount to the same machine integer as the value.
        const auto rhs = rewriter
                             .create<index::CastUOp>(
                                 op.getLoc(),
                                 adaptor.getValue().getType(),
                                 amount)
                             .getResult();

        // Create the LLVM shift operation.
        rewriter.replaceOpWithNewOp<To>(op, adaptor.getValue(), rhs);
        return success();
    }
};

using ConvertShl = ConvertShiftOp<ShlOp, LLVM::ShlOp>;
using ConvertShr = ConvertShiftOp<ShrOp, LLVM::LShrOp>;

struct ConvertCount : ConvertOpToLLVMPattern<CountOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(
        CountOp op,
        CountOp::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        const auto result =
            rewriter.create<LLVM::CtPopOp>(op.getLoc(), adaptor.getValue())
                .getResult();
        rewriter.replaceOpWithNewOp<index::CastUOp>(op, op.getType(), result);
        return success();
    }
};

template<class From, class To>
struct ConvertScan : ConvertOpToLLVMPattern<From> {
    using ConvertOpToLLVMPattern<From>::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(
        From op,
        typename From::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        const auto flag = rewriter
                              .create<LLVM::ConstantOp>(
                                  op.getLoc(),
                                  rewriter.getBoolAttr(false))
                              .getResult();
        const auto result = rewriter
                                .create<To>(
                                    op.getLoc(),
                                    this->getTypeConverter()->getIndexType(),
                                    adaptor.getValue(),
                                    rewriter.getIntegerAttr(flag.getType(), 1))
                                .getResult();
        rewriter.replaceOpWithNewOp<index::CastUOp>(op, op.getType(), result);
        return success();
    }
};

using ConvertClz = ConvertScan<ClzOp, LLVM::CountLeadingZerosOp>;
using ConvertCtz = ConvertScan<CtzOp, LLVM::CountTrailingZerosOp>;

} // namespace

void ConvertBitToLLVMPass::runOnOperation()
{
    // Set LLVM lowering options.
    LowerToLLVMOptions options(&getContext());
    if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
        options.overrideIndexBitwidth(indexBitwidth);
    LLVMTypeConverter typeConverter(&getContext(), options);

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    // NOTE: The LLVMTypeConverter can already convert to signless integers, but
    //       we must respect our BitSequenceType interface. Additionally, we do
    //       not need to exclude ShapedTypes that are not 1D vectors, since that
    //       is also already performed by the LLVMTypeConverter.

    // Add the custom signless integer materializer.
    typeConverter.addConversion(
        [](BitSequenceLikeType type) -> std::optional<Type> {
            const auto machineTy = IntegerType::get(
                type.getContext(),
                type.getElementType().getBitWidth());

            if (const auto shaped = type.dyn_cast<ShapedType>())
                return shaped.cloneWith(std::nullopt, machineTy);
            return machineTy;
        });
    const auto addBitCast = [&](OpBuilder &builder,
                                Type type,
                                ValueRange inputs,
                                Location loc) -> std::optional<Value> {
        // Only work on BitSequenceLikeTypes.
        if (!type.isa<BitSequenceLikeType>()) return std::nullopt;
        assert(inputs.size() == 1);
        if (!inputs.front().getType().isa<BitSequenceLikeType>())
            return std::nullopt;

        // Make sure that the result operation is properly converted.
        // BUG: Why does this not happen automatically when creating a CastOp?
        return ConvertCast::materialize(
            builder,
            typeConverter,
            loc,
            type,
            inputs.front());
    };
    typeConverter.addSourceMaterialization(addBitCast);
    typeConverter.addTargetMaterialization(addBitCast);

    // BUG: Add the funnel shift intrinsics to LLVM and remove this workaround.
    bit::populateLowerFunnelShiftPatterns(patterns);

    // Conversion goes via our own patterns and the index dialect.
    bit::populateConvertBitToLLVMPatterns(typeConverter, patterns);
    index::populateIndexToLLVMConversionPatterns(typeConverter, patterns);

    // All operations must be converted to LLVM.
    target.addIllegalDialect<bit::BitDialect>();
    target.addIllegalDialect<index::IndexDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addLegalDialect<LLVM::LLVMDialect>();

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns))))
        signalPassFailure();
}

void mlir::bit::populateConvertBitToLLVMPatterns(
    LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns)
{
    patterns.add<
        ConvertConstant,
        ConvertCast,
        ConvertCmp,
        ConvertSelect,
        ConvertAnd,
        ConvertOr,
        ConvertXor,
        ConvertShl,
        ConvertShr,
        ConvertCount,
        ConvertClz,
        ConvertCtz>(typeConverter);
}

std::unique_ptr<Pass> mlir::createConvertBitToLLVMPass()
{
    return std::make_unique<ConvertBitToLLVMPass>();
}
