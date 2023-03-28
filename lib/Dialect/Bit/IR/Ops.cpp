/// Implements the Bit dialect ops.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "base2-mlir/Dialect/Bit/IR/Ops.h"

#include "base2-mlir/Dialect/Bit/Analysis/BitFolder.h"
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
    Optional<Location>,
    ValueRange,
    DictionaryAttr attributes,
    RegionRange,
    SmallVectorImpl<Type> &result)
{
    const auto value = attributes.getAs<ValueLikeAttr>(getAttributeNames()[0]);
    if (!value) return failure();

    result.push_back(value.getType());
    return success();
}

OpFoldResult ConstantOp::fold(ConstantOp::FoldAdaptor adaptor)
{
    return adaptor.getValue();
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs)
{
    assert(inputs.size() == 1 && outputs.size() == 1);

    const auto inTy =
        getElementTypeOrSelf(inputs[0]).dyn_cast<BitSequenceType>();
    const auto outTy =
        getElementTypeOrSelf(outputs[0]).dyn_cast<BitSequenceType>();
    if (!inTy || !outTy) return false;

    return inTy.getBitWidth() == outTy.getBitWidth();
}

OpFoldResult CastOp::fold(CastOp::FoldAdaptor adaptor)
{
    // Fold no-op casts.
    if (getType() == getIn().getType()) return getIn();

    // Fold constant bit casts.
    if (const auto attr = adaptor.getIn().dyn_cast_or_null<ValueLikeAttr>())
        return BitFolder::bitCast(attr, getType());

    // Otherwise folding is not performed.
    return OpFoldResult{};
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
    const auto makeSplat = [&](bool value) -> Attribute {
        if (const auto shaped = getType().dyn_cast<ShapedType>())
            return DenseBitSequencesAttr::get(shaped, value);

        return ValueAttr::get(getType(), value);
    };

    // Fold trivial predicate.
    switch (getPredicate()) {
    case EqualityPredicate::Verum: return makeSplat(true);
    case EqualityPredicate::Falsum: return makeSplat(false);
    default: break;
    }

    // Fold trivial equality.
    if (getLhs() == getRhs()) return makeSplat(matches(true, getPredicate()));

    // Fold constant operands.
    const auto lhsAttr = adaptor.getLhs().dyn_cast_or_null<ValueLikeAttr>();
    const auto rhsAttr = adaptor.getRhs().dyn_cast_or_null<ValueLikeAttr>();
    if (lhsAttr && rhsAttr)
        return BitFolder::bitCmp(getPredicate(), lhsAttr, rhsAttr);

    // Otherwise folding is not performed.
    return OpFoldResult{};
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

        // Look for an operation on booleans with a right-hand constant.
        if (!op.getLhs().getType().getElementType().isSignlessInteger(1))
            return failure();
        const auto rhsAttr = getConstantValue(op.getRhs());
        if (!rhsAttr) return failure();

        // Determine the splat value of the right-hand constant.
        bool rhsVal;
        if (const auto rhsDense = rhsAttr.dyn_cast<DenseBitSequencesAttr>()) {
            if (!rhsDense.isSplat()) {
                // We could technically break apart this operation, but we
                // leave that to a later pass after de-tensorization.
                return failure();
            }
            rhsVal = rhsDense.getSplatValue().isOnes();
        } else {
            rhsVal = rhsAttr.cast<ValueAttr>().getValue().isOnes();
        }

        if ((op.getPredicate() == EqualityPredicate::Equal) == rhsVal) {
            // Operation is useless.
            rewriter.replaceOp(op, op.getLhs());
        } else {
            // Operation is a bitwise complement.
            const auto mask =
                rewriter
                    .create<ConstantOp>(
                        op.getLoc(),
                        ValueLikeAttr::getSplat(op.getRhs().getType(), true))
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
    if (const auto condType = getCondition().getType().dyn_cast<ShapedType>())
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
    // Fold trivial equality.
    if (getTrueValue() == getFalseValue()) return getTrueValue();

    // Fold constant conditionals.
    if (const auto cond =
            adaptor.getCondition().dyn_cast_or_null<ValueLikeAttr>()) {
        return BitFolder::bitSelect(
            cond,
            combine(adaptor.getTrueValue(), getTrueValue()),
            combine(adaptor.getFalseValue(), getFalseValue()));
    }

    // Otherwise no folding is performed.
    return OpFoldResult{};
}

//===----------------------------------------------------------------------===//
// Logic operations
//===----------------------------------------------------------------------===//

OpFoldResult AndOp::fold(AndOp::FoldAdaptor adaptor)
{
    // NOTE: Idempotency is folded by trait.

    // Fold if at least one operand is constant (commutative!).
    if (const auto attr = adaptor.getRhs().dyn_cast_or_null<ValueLikeAttr>())
        return BitFolder::bitAnd(combine(adaptor.getLhs(), getLhs()), attr);

    // Otherwise no folding is performed.
    return OpFoldResult{};
}

OpFoldResult OrOp::fold(OrOp::FoldAdaptor adaptor)
{
    // NOTE: Idempotency is folded by trait.

    // Fold if at least one operand is constant (commutative!).
    if (const auto attr = adaptor.getRhs().dyn_cast_or_null<ValueLikeAttr>())
        return BitFolder::bitOr(combine(adaptor.getLhs(), getLhs()), attr);

    // Otherwise no folding is performed.
    return OpFoldResult{};
}

OpFoldResult XorOp::fold(XorOp::FoldAdaptor adaptor)
{
    // Fold trivial equality.
    if (getLhs() == getRhs()) {
        return ValueLikeAttr::getSplat(
            getType(),
            BitSequence::zeros(getType().getElementType().getBitWidth()));
    }

    // Fold if at least one operand is constant (commutative!).
    if (const auto attr = adaptor.getRhs().dyn_cast_or_null<ValueLikeAttr>())
        return BitFolder::bitXor(combine(adaptor.getLhs(), getLhs()), attr);

    // Otherwise no folding is performed.
    return OpFoldResult{};
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
        const auto funnel = op.getFunnel();
        if (!funnel) return failure();
        const auto funnelAttr =
            cast_or_null<ValueAttr>(getConstantValue(funnel));
        if (!funnelAttr || !funnelAttr.getValue().isZeros()) return failure();

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
        const auto amountAttr = getConstantValue(op.getAmount())
                                    .template dyn_cast_or_null<IntegerAttr>();
        const auto srcAmountAttr =
            getConstantValue(src.getAmount())
                .template dyn_cast_or_null<IntegerAttr>();
        if (!amountAttr || !srcAmountAttr) return failure();

        // Operate on well-formed rotations only.
        const auto bitWidth = op.getType()
                                  .template cast<BitSequenceLikeType>()
                                  .getElementType()
                                  .getBitWidth();
        const auto amount = amountAttr.getValue().getZExtValue();
        const auto srcAmount = srcAmountAttr.getValue().getZExtValue();
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
    patterns.add<ZeroFunnel<ShlOp>, BalanceRotations<ShlOp, ShrOp>>(context);
}

OpFoldResult ShlOp::fold(ShlOp::FoldAdaptor adaptor)
{
    // Shift amount must be known to perform folding.
    const auto amountAttr = adaptor.getAmount().dyn_cast_or_null<IntegerAttr>();
    if (!amountAttr) return OpFoldResult{};
    assert(amountAttr.getType().isIndex());
    const auto amount = amountAttr.getValue().getZExtValue();
    if (amount > max_bit_width) return OpFoldResult{};

    // Delegate to BitFolder.
    return BitFolder::bitShl(
        combine(adaptor.getValue(), getValue()),
        static_cast<bit_width_t>(amount),
        combine(adaptor.getFunnel(), getFunnel()));
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
    // Shift amount must be known to perform folding.
    const auto amountAttr = adaptor.getAmount().dyn_cast_or_null<IntegerAttr>();
    if (!amountAttr) return OpFoldResult{};
    assert(amountAttr.getType().isIndex());
    const auto amount = amountAttr.getValue().getZExtValue();
    if (amount > max_bit_width) return OpFoldResult{};

    // Delegate to BitFolder.
    return BitFolder::bitShr(
        combine(adaptor.getValue(), getValue()),
        static_cast<bit_width_t>(amount),
        combine(adaptor.getFunnel(), getFunnel()));
}

void ShrOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns,
    MLIRContext* context)
{
    patterns.add<ZeroFunnel<ShrOp>, BalanceRotations<ShrOp, ShlOp>>(context);
}

//===----------------------------------------------------------------------===//
// Scanning operations
//===----------------------------------------------------------------------===//

OpFoldResult CountOp::fold(CountOp::FoldAdaptor adaptor)
{
    // Fold if operand is constant.
    if (const auto attr = adaptor.getValue().dyn_cast_or_null<ValueAttr>())
        return BitFolder::bitCount(attr);

    return OpFoldResult{};
}

OpFoldResult ClzOp::fold(ClzOp::FoldAdaptor adaptor)
{
    // Fold if operand is constant.
    if (const auto attr = adaptor.getValue().dyn_cast_or_null<ValueAttr>())
        return BitFolder::bitClz(attr);

    return OpFoldResult{};
}

OpFoldResult CtzOp::fold(CtzOp::FoldAdaptor adaptor)
{
    // Fold if operand is constant.
    if (const auto attr = adaptor.getValue().dyn_cast_or_null<ValueAttr>())
        return BitFolder::bitCtz(attr);

    return OpFoldResult{};
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
