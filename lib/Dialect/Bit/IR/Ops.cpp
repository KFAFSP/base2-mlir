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
    if (const auto attr =
            adaptor.getIn().dyn_cast_or_null<BitSequenceLikeAttr>())
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

        return BitSequenceAttr::get(getType(), value);
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
    const auto lhsAttr =
        adaptor.getLhs().dyn_cast_or_null<BitSequenceLikeAttr>();
    const auto rhsAttr =
        adaptor.getRhs().dyn_cast_or_null<BitSequenceLikeAttr>();
    if (lhsAttr && rhsAttr)
        return BitFolder::bitCmp(getPredicate(), lhsAttr, rhsAttr);

    // Otherwise folding is not performed.
    return OpFoldResult{};
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

void SelectOp::print(OpAsmPrinter &p)
{
    // $condition `,` $trueValue `,` $falseValue
    p << " ";
    p.printOperands(getOperands());

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
            adaptor.getCondition().dyn_cast_or_null<BitSequenceLikeAttr>()) {
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
    // Fold trivial equality.
    if (getLhs() == getRhs()) return getLhs();

    // Fold if at least one operand is constant (commutative!).
    if (const auto attr =
            adaptor.getRhs().dyn_cast_or_null<BitSequenceLikeAttr>())
        return BitFolder::bitAnd(combine(adaptor.getLhs(), getLhs()), attr);

    // Otherwise no folding is performed.
    return OpFoldResult{};
}

OpFoldResult OrOp::fold(OrOp::FoldAdaptor adaptor)
{
    // Fold trivial equality.
    if (getLhs() == getRhs()) return getLhs();

    // Fold if at least one operand is constant (commutative!).
    if (const auto attr =
            adaptor.getRhs().dyn_cast_or_null<BitSequenceLikeAttr>())
        return BitFolder::bitOr(combine(adaptor.getLhs(), getLhs()), attr);

    // Otherwise no folding is performed.
    return OpFoldResult{};
}

OpFoldResult XorOp::fold(XorOp::FoldAdaptor adaptor)
{
    // Fold trivial equality.
    if (getLhs() == getRhs()) {
        const auto result =
            BitSequence::zeros(getType().getElementType().getBitWidth());
        if (const auto shaped = getType().dyn_cast<ShapedType>()) {
            return DenseBitSequencesAttr::get(
                shaped.cloneWith(std::nullopt, getType().getElementType()),
                result);
        }

        return BitSequenceAttr::get(getType().getElementType(), result);
    }

    // Fold if at least one operand is constant (commutative!).
    if (const auto attr =
            adaptor.getRhs().dyn_cast_or_null<BitSequenceLikeAttr>())
        return BitFolder::bitXor(combine(adaptor.getLhs(), getLhs()), attr);

    // Otherwise no folding is performed.
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
