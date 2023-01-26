/// Implements the Base2 dialect fixed-point ops.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "ShapedUtils.h"
#include "base2-mlir/Dialect/Base2/Analysis/FixedPointInterpreter.h"
#include "base2-mlir/Dialect/Base2/IR/Ops.h"
#include "mlir/IR/OpImplementation.h"

#include <numeric>

#define DEBUG_TYPE "base2-fixed-ops"

using namespace mlir;
using namespace mlir::base2;

//===----------------------------------------------------------------------===//
// FixedPromoteOp
//===----------------------------------------------------------------------===//

ParseResult FixedPromoteOp::parse(OpAsmParser &p, OperationState &result)
{
    // operands
    SmallVector<OpAsmParser::UnresolvedOperand> operands;
    if (p.parseOperandList(operands, 1)) return failure();
    if (p.parseColon()) return failure();

    // `:` type(operands) `to` $resultType
    SmallVector<Type> operandTys;
    if (p.parseTypeList(operandTys)) return failure();
    if (p.parseKeyword("to")) return failure();
    Type resultTy;
    if (p.parseType(resultTy)) return failure();

    // Resolve operands.
    if (p.resolveOperands(
            operands,
            operandTys,
            p.getNameLoc(),
            result.operands))
        return failure();

    // Construct result types.
    for (auto opTy : operandTys)
        result.types.push_back(getSameShape(opTy, resultTy));
    return success();
}

void FixedPromoteOp::print(OpAsmPrinter &p)
{
    // operands
    p.printOperands(getOperands());
    // `:` type(operands)
    p << " : ";
    llvm::interleaveComma(getOperandTypes(), p);
    // `to` $resultType
    p << " to ";
    p << getElementTypeOrSelf(getResultTypes().front());
}

LogicalResult FixedPromoteOp::verify()
{
    // At least one operand must be given.
    if (getNumOperands() == 0)
        return emitOpError() << "expected 1 or more operands";

    // Number of operands and results must match.
    if (getNumOperands() != getNumResults())
        return emitOpError() << "number of results (" << getNumResults()
                             << ") does not match number of operands ("
                             << getNumOperands() << ")";

    // All results must:
    const auto resultTy = getElementTypeOrSelf(getResultTypes().front());
    const auto resultSema = resultTy.cast<FixedPointSemantics>();
    for (auto [idx, resTy] : llvm::enumerate(getResultTypes())) {
        const auto opTy = getOperandTypes()[idx];

        // - have the same element type
        if (getElementTypeOrSelf(resTy) != resultTy)
            return emitOpError()
                   << "result #" << idx << " has incompatible type (" << resTy
                   << " != " << resultTy << ")";

        // - have the same shape as their operand
        if (failed(verifyCompatibleShapes({opTy, resTy})))
            return emitOpError()
                   << "result #" << idx << " has incompatible shape (" << resTy
                   << " !~ " << opTy << ")";

        // - be a supertype of their operand
        const auto opSema = getOperandTypes()[idx].cast<FixedPointSemantics>();
        if (!resultSema.isSupersetOf(opSema))
            return emitOpError() << "operand #" << idx << " is not a subtype ("
                                 << opTy << " > " << resTy << ")";
    }

    return success();
}

void FixedPromoteOp::build(
    OpBuilder &builder,
    OperationState &state,
    Value value,
    FixedPointSemantics resultTy)
{
    assert(value);
    if (!resultTy) {
        // Infer result type to input type.
        resultTy =
            getElementTypeOrSelf(value.getType()).cast<FixedPointSemantics>();
    }

    build(builder, state, {getSameShape(value.getType(), resultTy)}, {value});
}

void FixedPromoteOp::build(
    OpBuilder &builder,
    OperationState &state,
    ValueRange values,
    FixedPointSemantics resultTy)
{
    assert(!values.empty());
    if (!resultTy) {
        // Infer the result type.
        const auto semas = llvm::map_range(
            values,
            [](Value val) {
                return getElementTypeOrSelf(val.getType())
                    .cast<FixedPointSemantics>();
            });
        resultTy = std::accumulate(
            std::next(semas.begin()),
            semas.end(),
            *semas.begin(),
            [](auto l, auto r) {
                return FixedPointInterpreter::promote(l, r);
            });
    }

    build(
        builder,
        state,
        llvm::to_vector(llvm::map_range(
            values.getTypes(),
            [&](Type opTy) { return getSameShape(opTy, resultTy); })),
        values);
}

LogicalResult FixedPromoteOp::fold(
    FixedPromoteOp::FoldAdaptor adaptor,
    SmallVectorImpl<OpFoldResult> &results)
{
    const auto resultTys = getResultTypes();
    const auto resultTy = getElementTypeOrSelf(resultTys[0]);
    const auto resultSema = resultTy.cast<FixedPointSemantics>();

    auto folded = false;
    for (auto [idx, op] : llvm::enumerate(getOperands())) {
        const auto opTy = getElementTypeOrSelf(op.getType());
        const auto opSema = opTy.cast<FixedPointSemantics>();

        // Try constant folding.
        if (const auto bits = adaptor.getOperands()[idx]
                                  .dyn_cast_or_null<BitSequenceLikeAttr>()) {
            results.push_back(bits.map(
                [&](const BitSequence &value) {
                    return FixedPointInterpreter::valueCast(
                               opSema,
                               value,
                               resultSema)
                        .value();
                },
                getSameShape(op.getType(), resultTy)));
            folded |= true;
            continue;
        }

        // Try no-op folding.
        if (op.getType() == resultTys[idx]) {
            results.push_back(op);
            folded |= true;
        }
    }

    return success(folded);
}
