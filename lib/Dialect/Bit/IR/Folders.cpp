/// Implements the Bit dialect folding helper.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "base2-mlir/Dialect/Bit/IR/Folders.h"
#include "base2-mlir/Dialect/Bit/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeUtilities.h"
#include "ub-mlir/Dialect/UB/Utils/StaticFolder.h"

using namespace mlir;
using namespace mlir::bit;
using namespace mlir::bit::match;
using namespace mlir::ub;
using namespace mlir::ub::match;

//===----------------------------------------------------------------------===//
// BitFolder
//===----------------------------------------------------------------------===//

OpFoldResult BitFolder::bitCast(CastOp op, ArrayRef<Attribute> operands)
{
    static const StaticFolder folder(
        [](CastOp op, AnyPoison) { return ConstOrPoison::get(op.getType()); },
        [](CastOp op, Const in) {
            return in.bitCastElements(op.getType().getElementType());
        },
        [](CastOp op, ConstOrPoison in) {
            return in.map(
                [](const auto &x) { return x; },
                op.getType().getElementType());
        });

    return folder(op, operands);
}

OpFoldResult BitFolder::bitCmp(CmpOp op, ArrayRef<Attribute> operands)
{
    static const StaticFolder folder(
        [](CmpOp op, ValueRange) -> OpFoldResult {
            const auto type = llvm::cast<BitSequenceLikeType>(op.getType());
            switch (op.getPredicate()) {
            case EqualityPredicate::Verum: return Const::getSplat(type, true);
            case EqualityPredicate::Falsum: return Const::getSplat(type, false);
            default: return {};
            }
        },
        [](CmpOp op, Value, AnyPoison) {
            return PoisonAttr::get(op.getType());
        },
        [](CmpOp op, AnyWellDefined lhs, Any rhs) -> OpFoldResult {
            if (lhs == rhs)
                return Const::getSplat(
                    llvm::cast<BitSequenceLikeType>(op.getType()),
                    matches(true, op.getPredicate()));
            return {};
        },
        [](CmpOp op, ConstOrPoison lhs, ConstOrPoison rhs) {
            return lhs.zip(
                [pred = op.getPredicate()](const auto &l, const auto &r)
                    -> std::optional<BitSequence> {
                    if (!l || !r) return poison;
                    return matches(*l == *r, pred);
                },
                rhs,
                IntegerType::get(op.getContext(), 1)
                    .dyn_cast<BitSequenceType>());
        });

    return folder(op, operands);
}

OpFoldResult BitFolder::bitSelect(SelectOp op, ArrayRef<Attribute> operands)
{
    static const StaticFolder folder(
        [](Any, Any trueValue, Any falseValue) -> OpFoldResult {
            if (trueValue == falseValue) return trueValue;
            return {};
        },
        [](AnyPoison, Type type, Value) {
            return ConstOrPoison::get(llvm::cast<BitSequenceLikeType>(type));
        },
        [](Zeros, Any, Any falseValue) { return falseValue; },
        [](Ones, Any trueValue, Any) { return trueValue; },
        [](ConstOrPoison cond,
           ConstOrPoison trueValue,
           ConstOrPoison falseValue) {
            return cond.zip(
                [](const auto &c,
                   const auto &t,
                   const auto &f) -> std::optional<BitSequence> {
                    if (!c) return poison;
                    return c->isOnes() ? t : f;
                },
                trueValue,
                falseValue,
                trueValue.getType().getElementType());
        });

    return folder(op, operands);
}

OpFoldResult BitFolder::bitAnd(AndOp op, ArrayRef<Attribute> operands)
{
    static const StaticFolder folder(
        [](Any lhs, Any rhs) -> OpFoldResult {
            if (lhs == rhs) return lhs;
            return {};
        },
        [](Any lhs, Ones) { return lhs; },
        [](Any, Zeros rhs) { return rhs; },
        [](ConstOrPoison lhs, ConstOrPoison rhs) {
            return lhs.zip(
                [](const auto &l, const auto &r) -> std::optional<BitSequence> {
                    if (l && l->isZeros()) return *l;
                    if (r && r->isZeros()) return *r;
                    if (!l || !r) return poison;
                    return l->logicAnd(*r);
                },
                rhs);
        });

    return folder(op, operands);
}

OpFoldResult BitFolder::bitOr(OrOp op, ArrayRef<Attribute> operands)
{
    static const StaticFolder folder(
        [](Any lhs, Any rhs) -> OpFoldResult {
            if (lhs == rhs) return lhs;
            return {};
        },
        [](Any, Ones rhs) { return rhs; },
        [](Any lhs, Zeros) { return lhs; },
        [](ConstOrPoison lhs, ConstOrPoison rhs) {
            return lhs.zip(
                [](const auto &l, const auto &r) -> std::optional<BitSequence> {
                    if (l && l->isOnes()) return *l;
                    if (r && r->isOnes()) return *r;
                    if (!l || !r) return poison;
                    return l->logicOr(*r);
                },
                rhs);
        });

    return folder(op, operands);
}

OpFoldResult BitFolder::bitXor(XorOp op, ArrayRef<Attribute> operands)
{
    static const StaticFolder folder(
        [](XorOp op, Any lhs, Any rhs) -> OpFoldResult {
            if (lhs == rhs) {
                const auto type = llvm::cast<BitSequenceLikeType>(op.getType());
                return Const::getSplat(
                    type,
                    BitSequence::zeros(type.getElementType().getBitWidth()));
            }
            return {};
        },
        [](Any lhs, Zeros) { return lhs; },
        [](ConstOrPoison lhs, ConstOrPoison rhs) {
            return lhs.zip(
                [](const auto &l, const auto &r) -> std::optional<BitSequence> {
                    if (!l || !r) return poison;
                    return l->logicXor(*r);
                },
                rhs);
        });

    return folder(op, operands);
}

// NOTE: Both shift ops have the same layout of their operands, regardless of
//       how value and funnel are logically arranged.
static const StaticFolder shiftFolderBase(
    [](ShiftOp op, Any, Poison<IndexType>) {
        return ConstOrPoison::get(op.getResult().getType());
    },
    [](ShiftOp op, Any, Poison<IndexType>, Any) {
        return ConstOrPoison::get(op.getResult().getType());
    },
    [](ShiftOp op, Any value, ConstIndex amount) -> OpFoldResult {
        const auto type = op.getResult().getType();
        const auto bitWidth = type.getElementType().getBitWidth();

        if (amount.getValue() == 0) return value;
        if (amount >= bitWidth)
            return Const::getSplat(type, BitSequence::zeros(bitWidth));
        return {};
    },
    [](ShiftOp op, Any value, ConstIndex amount, Any) -> OpFoldResult {
        const auto type = op.getResult().getType();
        const auto bitWidth = type.getElementType().getBitWidth();

        if (amount.getValue() == 0) return value;
        if (amount >= 2 * bitWidth)
            return Const::getSplat(type, BitSequence::zeros(bitWidth));
        return {};
    },
    [](ShiftOp op, AnyPoison, ConstIndex amount, Any) -> OpFoldResult {
        const auto type = op.getResult().getType();
        const auto bitWidth = type.getElementType().getBitWidth();

        if (amount < bitWidth) return ConstOrPoison::get(type);
        return {};
    },
    [](Any, Any, AnyPoison funnel) { return funnel; });

OpFoldResult BitFolder::bitShl(ShlOp op, ArrayRef<Attribute> operands)
{
    static const StaticFolder folder(
        shiftFolderBase,
        [](ConstOrPoison value, ConstIndex amount) {
            return value.map(
                [amount = amount.getValue()](
                    const auto &el) -> std::optional<BitSequence> {
                    if (!el) return poison;
                    return el->logicShl(amount);
                });
        },
        [](ShlOp op,
           ConstOrPoison value,
           ConstIndex amount,
           ConstOrPoison funnel) {
            const auto type = op.getResult().getType();
            const auto bitWidth = type.getElementType().getBitWidth();
            return value.zip(
                [bitWidth = bitWidth, amount = amount.getValue()](
                    const auto &v,
                    const auto &f) -> std::optional<BitSequence> {
                    if (!f) return poison;
                    if (!v) {
                        if (amount < bitWidth) return poison;
                        return f->logicShl(amount - bitWidth);
                    }
                    return v->funnelShl(*f, amount);
                },
                funnel);
        });

    return folder(op, operands);
}

OpFoldResult BitFolder::bitShr(ShrOp op, ArrayRef<Attribute> operands)
{
    static const StaticFolder folder(
        shiftFolderBase,
        [](ConstOrPoison value, ConstIndex amount) {
            return value.map(
                [amount = amount.getValue()](
                    const auto &el) -> std::optional<BitSequence> {
                    if (!el) return poison;
                    return el->logicShr(amount);
                });
        },
        [](ShrOp op,
           ConstOrPoison funnel,
           ConstIndex amount,
           ConstOrPoison value) {
            const auto type = op.getResult().getType();
            const auto bitWidth = type.getElementType().getBitWidth();
            return value.zip(
                [bitWidth = bitWidth, amount = amount.getValue()](
                    const auto &v,
                    const auto &f) -> std::optional<BitSequence> {
                    if (!f) return poison;
                    if (!v) {
                        if (amount < bitWidth) return poison;
                        return f->logicShr(amount - bitWidth);
                    }
                    return v->funnelShr(*f, amount);
                },
                funnel);
        });

    return folder(op, operands);
}

static const StaticFolder countFolderBase([](AnyPoison operand) {
    return PoisonAttr::get(IndexType::get(operand.getContext()));
});

OpFoldResult BitFolder::bitCount(CountOp op, ArrayRef<Attribute> operands)
{
    static const StaticFolder folder(
        countFolderBase,
        [](BitSequenceAttr operand) {
            return IntegerAttr::get(
                IndexType::get(operand.getContext()),
                operand.getValue().countOnes());
        });

    return folder(op, operands);
}

OpFoldResult BitFolder::bitClz(ClzOp op, ArrayRef<Attribute> operands)
{
    static const StaticFolder folder(
        countFolderBase,
        [](BitSequenceAttr operand) {
            return IntegerAttr::get(
                IndexType::get(operand.getContext()),
                operand.getValue().countLeadingZeros());
        });

    return folder(op, operands);
}

OpFoldResult BitFolder::bitCtz(CtzOp op, ArrayRef<Attribute> operands)
{
    static const StaticFolder folder(
        countFolderBase,
        [](BitSequenceAttr operand) {
            return IntegerAttr::get(
                IndexType::get(operand.getContext()),
                operand.getValue().countTrailingZeros());
        });

    return folder(op, operands);
}
