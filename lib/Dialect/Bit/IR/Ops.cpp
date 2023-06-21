/// Implements the Bit dialect ops.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "base2-mlir/Dialect/Bit/IR/Folders.h"
#include "base2-mlir/Dialect/Bit/IR/Matchers.h"
#include "base2-mlir/Dialect/Bit/IR/Ops.h"
#include "base2-mlir/Utils.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

#define DEBUG_TYPE "bit-ops"

using namespace mlir;
using namespace mlir::bit;
using namespace mlir::ext;

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "base2-mlir/Dialect/Bit/IR/Ops.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

LogicalResult ConstantOp::inferReturnTypes(
    MLIRContext*,
    std::optional<Location>,
    ValueRange,
    DictionaryAttr attributes,
    OpaqueProperties,
    RegionRange,
    SmallVectorImpl<Type> &result)
{
    const auto value =
        attributes.getAs<BitSequenceLikeAttr>(getAttributeNames()[0]);
    if (!value) return failure();

    result.push_back(value.getType());
    return success();
}

OpFoldResult ConstantOp::fold(ConstantOp::FoldAdaptor) { return getValue(); }

namespace {

struct BuiltinConstant : OpRewritePattern<ConstantOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(ConstantOp op, PatternRewriter &rewriter) const override
    {
        // Construct the canonical attribute.
        const auto canonical = BitSequenceLikeAttr::get(op.getValue());

        // Attempt to update the value.
        if (op.getValue() == canonical) return failure();
        rewriter.updateRootInPlace(op, [&]() { op.setValueAttr(canonical); });
        return success();
    }
};

} // namespace

void ConstantOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns,
    MLIRContext* context)
{
    patterns.add<BuiltinConstant>(context);
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs)
{
    assert(inputs.size() == 1 && outputs.size() == 1);

    const auto inTy =
        llvm::dyn_cast<BitSequenceType>(getElementTypeOrSelf(inputs[0]));
    const auto outTy =
        llvm::dyn_cast<BitSequenceType>(getElementTypeOrSelf(outputs[0]));
    if (!inTy || !outTy) return false;

    return inTy.getBitWidth() == outTy.getBitWidth();
}

OpFoldResult CastOp::fold(CastOp::FoldAdaptor adaptor)
{
    return BitFolder::bitCast(*this, adaptor.getOperands());
}

namespace {

// bitcast(bitcast(x : A) : B) -> bitcast(x : B)
struct TransitiveBitCastPattern : OpRewritePattern<CastOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(CastOp op, PatternRewriter &rewriter) const override
    {
        // Find the topmost CastOp in this tree.
        auto root = op;
        while (const auto parent = root.getIn().getDefiningOp<CastOp>())
            root = parent;

        // No change possible.
        if (op == root) return failure();

        // Directly cast from root input to desired output.
        rewriter.updateRootInPlace(op, [&]() {
            op.setOperand(root.getOperand());
        });
        return success();
    }
};

} // namespace

void CastOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns,
    MLIRContext* context)
{
    patterns.add<TransitiveBitCastPattern>(context);
}

//===----------------------------------------------------------------------===//
// CmpOp
//===----------------------------------------------------------------------===//

OpFoldResult CmpOp::fold(CmpOp::FoldAdaptor adaptor)
{
    return BitFolder::bitCmp(*this, adaptor.getOperands());
}

namespace {

struct BooleanComparison : OpRewritePattern<CmpOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(CmpOp op, PatternRewriter &rewriter) const override
    {
        // Ignore trivial cases.
        switch (op.getPredicate()) {
        case EqualityPredicate::Verum:
        case EqualityPredicate::Falsum: return failure();
        default: break;
        }

        // Look for an operation on booleans with a right-hand splat constant.
        if (!op.getLhs().getType().getElementType().isSignlessInteger(1))
            return failure();
        const auto rhs =
            llvm::dyn_cast_or_null<match::Splat>(getConstantValue(op.getRhs()));
        if (!rhs) return failure();

        if ((op.getPredicate() == EqualityPredicate::Equal)
            == rhs.getValue().isOnes()) {
            // Operation is useless.
            rewriter.replaceOp(op, op.getLhs());
        } else {
            // Operation is a bitwise complement.
            const auto mask = rewriter
                                  .create<ConstantOp>(
                                      op.getLoc(),
                                      BitSequenceLikeAttr::getSplat(
                                          op.getRhs().getType(),
                                          true))
                                  .getResult();
            rewriter.replaceOpWithNewOp<XorOp>(op, op.getLhs(), mask);
        }

        return success();
    }
};

} // namespace

void CmpOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns,
    MLIRContext* context)
{
    patterns.add<BooleanComparison>(context);
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

void SelectOp::print(OpAsmPrinter &p)
{
    // $condition `,` $trueValue `,` $falseValue
    p << " " << getOperands();

    // attr-dict
    p.printOptionalAttrDict((*this)->getAttrs());

    // `:`
    p << " : ";

    // [type($condition) `,`]
    if (const auto condType =
            llvm::dyn_cast<ShapedType>(getCondition().getType()))
        p << condType << ", ";

    // type($result)
    p << getType();
}

ParseResult SelectOp::parse(OpAsmParser &p, OperationState &result)
{
    // $condition `,` $trueValue `,` $falseValue
    SmallVector<OpAsmParser::UnresolvedOperand> operands;
    if (p.parseOperandList(operands, 3)) return failure();

    // attr-dict
    if (p.parseOptionalAttrDict(result.attributes)) return failure();

    // `:` [type($condition) `,`] type($result)
    SmallVector<Type> types;
    if (p.parseColonTypeList(types)) return failure();

    // Resolve the condition type.
    switch (types.size()) {
    case 1:
        if (p.resolveOperand(
                operands[0],
                p.getBuilder().getI1Type(),
                result.operands))
            return failure();
        break;

    case 2:
        if (p.resolveOperand(operands[0], types.front(), result.operands))
            return failure();
        break;

    default: return p.emitError(p.getNameLoc(), "expected 1 or 2 types");
    }

    // Resolve the operand and result types.
    result.addTypes(types.back());
    if (p.resolveOperand(operands[1], types.back(), result.operands))
        return failure();
    if (p.resolveOperand(operands[2], types.back(), result.operands))
        return failure();

    return success();
}

OpFoldResult SelectOp::fold(SelectOp::FoldAdaptor adaptor)
{
    return BitFolder::bitSelect(*this, adaptor.getOperands());
}

namespace {

struct BooleanSelect : OpRewritePattern<SelectOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(SelectOp op, PatternRewriter &rewriter) const override
    {
        // Look for select ops returning boolean(s).
        if (!op.getType().getElementType().isSignlessInteger(1))
            return failure();

        // Look for splat alternative operands.
        const auto trueSplat = llvm::dyn_cast_or_null<match::Splat>(
            getConstantValue(op.getTrueValue()));
        const auto falseSplat = llvm::dyn_cast_or_null<match::Splat>(
            getConstantValue(op.getFalseValue()));
        if (!trueSplat || !falseSplat) return failure();

        if (trueSplat.getValue().isOnes() && falseSplat.getValue().isZeros()) {
            // Trivial boolean pass-through.
            rewriter.replaceOp(op, op.getCondition());
            return success();
        }

        if (trueSplat.getValue().isZeros() && falseSplat.getValue().isOnes()) {
            // Bitwise complement.
            rewriter.replaceOpWithNewOp<XorOp>(
                op,
                op.getCondition(),
                rewriter
                    .create<ConstantOp>(
                        op.getLoc(),
                        match::Const::getSplat(
                            llvm::cast<BitSequenceLikeType>(
                                op.getCondition().getType()),
                            true))
                    .getResult());
            return success();
        }

        return failure();
    }
};

} // namespace

void SelectOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns,
    MLIRContext* context)
{
    patterns.add<BooleanSelect>(context);
}

//===----------------------------------------------------------------------===//
// Logic operations
//===----------------------------------------------------------------------===//

OpFoldResult AndOp::fold(AndOp::FoldAdaptor adaptor)
{
    return BitFolder::bitAnd(*this, adaptor.getOperands());
}

OpFoldResult OrOp::fold(OrOp::FoldAdaptor adaptor)
{
    return BitFolder::bitOr(*this, adaptor.getOperands());
}

OpFoldResult XorOp::fold(XorOp::FoldAdaptor adaptor)
{
    return BitFolder::bitXor(*this, adaptor.getOperands());
}

//===----------------------------------------------------------------------===//
// Shifting operations
//===----------------------------------------------------------------------===//

namespace {

template<class Op>
struct ZeroFunnel : OpRewritePattern<Op> {
    using OpRewritePattern<Op>::OpRewritePattern;

    LogicalResult
    matchAndRewrite(Op op, PatternRewriter &rewriter) const override
    {
        // Look for a constant zeros funnel.
        const auto funnelValue = op.getFunnel();
        if (!funnelValue) return failure();
        if (!llvm::isa_and_present<match::Zeros>(getConstantValue(funnelValue)))
            return failure();

        // Remove the funnel operand.
        const auto resultOp = rewriter.replaceOpWithNewOp<Op>(
            op,
            op.getValue(),
            op.getAmount(),
            Value{});
        resultOp->setAttrs(op->getAttrs());
        return success();
    }
};

template<class Op>
struct FunnelOnly : OpRewritePattern<Op> {
    using OpRewritePattern<Op>::OpRewritePattern;

    LogicalResult
    matchAndRewrite(Op op, PatternRewriter &rewriter) const override
    {
        // Look for funnel shifts with a constant amount.
        if (!op.getFunnel()) return failure();
        const auto amount = llvm::dyn_cast_or_null<match::ConstIndex>(
            getConstantValue(op.getAmount()));
        if (!amount) return failure();

        // Look for shift amounts that shift out the value.
        const auto bitWidth = op.getType().getElementType().getBitWidth();
        if (amount < bitWidth) return failure();

        // Remove the value operand.
        const auto resultOp = rewriter.replaceOpWithNewOp<Op>(
            op,
            op.getFunnel(),
            rewriter.create<index::ConstantOp>(op.getLoc(), amount - bitWidth)
                .getResult());
        resultOp->setAttrs(op->getAttrs());
        return success();
    }
};

template<class Op, class Inv>
struct BalanceRotations : OpRewritePattern<Op> {
    using OpRewritePattern<Op>::OpRewritePattern;

    LogicalResult
    matchAndRewrite(Op op, PatternRewriter &rewriter) const override
    {
        // Only applies to chained bit rotations.
        if (op.getValue() != op.getFunnel()) return failure();
        auto src = op.getValue().template getDefiningOp<ShiftOp>();
        if (!src || (src.getValue() != src.getFunnel())) return failure();

        // Remove trivial inverse rotations.
        if (op.getAmount() == src.getAmount()
            && op.getDirection() != src.getDirection()) {
            rewriter.replaceOp(op, src.getValue());
            return success();
        }

        // Operate on known shift amounts only.
        const auto amount = llvm::dyn_cast_or_null<match::ConstIndex>(
            getConstantValue(op.getAmount()));
        const auto srcAmount = llvm::dyn_cast_or_null<match::ConstIndex>(
            getConstantValue(src.getAmount()));
        if (!amount || !srcAmount) return failure();

        // Operate on well-formed rotations only.
        const auto bitWidth = llvm::cast<BitSequenceLikeType>(op.getType())
                                  .getElementType()
                                  .getBitWidth();
        if (amount > bitWidth || srcAmount > bitWidth) return failure();

        // Compute the rotation amount relative to the source.
        using amount_t = std::make_signed_t<bit_width_t>;
        auto totalAmount = static_cast<amount_t>(amount);
        if (op.getDirection() != src.getDirection())
            totalAmount -= srcAmount;
        else
            totalAmount += srcAmount;
        totalAmount %= bitWidth;

        // Handle trivial case.
        if (totalAmount == 0) {
            rewriter.replaceOp(op, src.getValue());
            return success();
        }

        // Generate shortened rotation.
        const auto totalAmountVal =
            rewriter
                .create<index::ConstantOp>(
                    op.getLoc(),
                    rewriter.getIndexAttr(
                        totalAmount < 0 ? -totalAmount : totalAmount))
                .getResult();
        if (totalAmount > 0)
            rewriter.replaceOpWithNewOp<Op>(
                op,
                src.getValue(),
                totalAmountVal,
                src.getValue());
        else
            rewriter.replaceOpWithNewOp<Inv>(
                op,
                src.getValue(),
                totalAmountVal,
                src.getValue());
        return success();
    }
};

} // namespace

void ShlOp::print(OpAsmPrinter &p)
{
    // $value
    p << " " << getValue();

    // (`:` $funnel)?
    if (auto funnel = getFunnel()) p << ":" << funnel;

    // `,` $amount
    p << ", " << getAmount();

    // attr-dict
    p.printOptionalAttrDict((*this)->getAttrs());

    // `:` type($result)
    p << " : " << getType();
}

ParseResult ShlOp::parse(OpAsmParser &p, OperationState &result)
{
    // $value
    OpAsmParser::UnresolvedOperand value, funnel, amount;
    if (p.parseOperand(value)) return failure();

    // (`:` $funnel)?
    auto hasFunnel = !p.parseOptionalColon();
    if (hasFunnel) {
        if (p.parseOperand(funnel)) return failure();
    }

    // `,` $amount
    if (p.parseComma()) return failure();
    if (p.parseOperand(amount)) return failure();

    // attr-dict
    if (p.parseOptionalAttrDict(result.attributes)) return failure();

    // `:` type($result)
    if (p.parseColon()) return failure();
    if (p.parseType(result.types.emplace_back())) return failure();

    // Resolve operands
    if (p.resolveOperand(value, result.types.front(), result.operands))
        return failure();
    if (p.resolveOperand(
            amount,
            p.getBuilder().getIndexType(),
            result.operands))
        return failure();
    if (hasFunnel) {
        if (p.resolveOperand(funnel, result.types.front(), result.operands))
            return failure();
    }

    return success();
}

LogicalResult ShlOp::verify()
{
    if (auto funnel = getFunnel()) {
        if (funnel.getType() != getType())
            return emitOpError()
                   << "funnel must have result type (" << funnel.getType()
                   << " != " << getType() << ")";
    }

    return success();
}

void ShlOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns,
    MLIRContext* context)
{
    patterns.add<
        ZeroFunnel<ShlOp>,
        FunnelOnly<ShlOp>,
        BalanceRotations<ShlOp, ShrOp>>(context);
}

OpFoldResult ShlOp::fold(ShlOp::FoldAdaptor adaptor)
{
    return BitFolder::bitShl(*this, adaptor.getOperands());
}

void ShrOp::print(OpAsmPrinter &p)
{
    p << " ";

    // ($funnel `:`)?
    if (auto funnel = getFunnel()) p << funnel << ":";

    // $value `,` $amount
    p << getValue() << ", " << getAmount();

    // attr-dict
    p.printOptionalAttrDict((*this)->getAttrs());

    // `:` type($result)
    p << " : " << getType();
}

ParseResult ShrOp::parse(OpAsmParser &p, OperationState &result)
{
    // ($funnel `:`)?
    OpAsmParser::UnresolvedOperand value, funnel, amount;
    if (p.parseOperand(value)) return failure();

    // $value
    auto hasFunnel = !p.parseOptionalColon();
    if (hasFunnel) {
        funnel = value;
        if (p.parseOperand(value)) return failure();
    }

    // `,` $amount
    if (p.parseComma()) return failure();
    if (p.parseOperand(amount)) return failure();

    // attr-dict
    if (p.parseOptionalAttrDict(result.attributes)) return failure();

    // `:` type($result)
    if (p.parseColon()) return failure();
    if (p.parseType(result.types.emplace_back())) return failure();

    // Resolve operands.
    if (p.resolveOperand(value, result.types.front(), result.operands))
        return failure();
    if (p.resolveOperand(
            amount,
            p.getBuilder().getIndexType(),
            result.operands))
        return failure();
    if (hasFunnel) {
        if (p.resolveOperand(funnel, result.types.front(), result.operands))
            return failure();
    }

    return success();
}

LogicalResult ShrOp::verify()
{
    if (auto funnel = getFunnel()) {
        if (funnel.getType() != getType())
            return emitOpError()
                   << "funnel must have result type (" << funnel.getType()
                   << " != " << getType() << ")";
    }

    return success();
}

OpFoldResult ShrOp::fold(ShrOp::FoldAdaptor adaptor)
{
    return BitFolder::bitShr(*this, adaptor.getOperands());
}

void ShrOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns,
    MLIRContext* context)
{
    patterns.add<
        ZeroFunnel<ShrOp>,
        FunnelOnly<ShrOp>,
        BalanceRotations<ShrOp, ShlOp>>(context);
}

//===----------------------------------------------------------------------===//
// Scanning operations
//===----------------------------------------------------------------------===//

OpFoldResult CountOp::fold(CountOp::FoldAdaptor adaptor)
{
    return BitFolder::bitCount(*this, adaptor.getOperands());
}

OpFoldResult ClzOp::fold(ClzOp::FoldAdaptor adaptor)
{
    return BitFolder::bitClz(*this, adaptor.getOperands());
}

OpFoldResult CtzOp::fold(CtzOp::FoldAdaptor adaptor)
{
    return BitFolder::bitCtz(*this, adaptor.getOperands());
}

//===----------------------------------------------------------------------===//
// BitDialect
//===----------------------------------------------------------------------===//

void BitDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "base2-mlir/Dialect/Bit/IR/Ops.cpp.inc"
        >();
}
