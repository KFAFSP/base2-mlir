//===- Ops.td ----------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the Base2 dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef BASE2_OPS
#define BASE2_OPS

include "base2-mlir/Dialect/Base2/IR/Types.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

def CastOp : Base2_Op<
        "cast",
        [
            Pure,
            SameOperandsAndResultShape,
            Base2_RoundingOp,
            DeclareOpInterfaceMethods<CastOpInterface>]> {
    let summary = "Value casting for interpretable types";
    let description = [{
        The `cast` operation obtains a value of a different type that
        represents the input value. If the result is not exactly representable,
        it follows the guarantee requested by the rounding mode.

        Syntax:

        ```
        cast-op ::= `cast` $in `:` type($in) `to` ($roundingMode)? type($out)
        ```

        Example:

        ```
        %0 = base2.cast %flt : f32 to nearest si32
        ```
    }];

    let arguments = (ins
        Base2_InterpretableLike:$in,
        Base2_RoundingModeDefaultAttr:$roundingMode);
    let results = (outs Base2_InterpretableLike:$out);

    let assemblyFormat = [{
        $in attr-dict `:` type($in) `to` ($roundingMode^)? type($out)
    }];

    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Common operations
//===----------------------------------------------------------------------===//

def Base2_CmpOp : Base2_Op<
        "cmp",
        [
            Pure,
            OpAsmOpInterface,
            SameTypeOperands,
            SameOperandsAndResultShape,
            TypesMatchWith<
                "result is i1 of operand shape",
                "lhs", "result", "::getI1SameShape($_self)">]
            # ElementwiseMappable.traits> {
    let summary = "Value comparison for interpretable bit sequences";
    let description = [{
        The `cmp` operation compares the two operands according to the specified
        predicate and returns a boolean or container of booleans indicating the
        result.

        Syntax:

        ```
        cmp-op ::= `cmp` $predicate $lhs, $rhs `:` type($lhs)
        ```

        Example:

        ```
        %0 = base2.cmp oeq %1, %2 : si16
        ```
    }];

    let arguments = (ins
        Base2_PartialOrderingPredicateAttr:$predicate,
        Base2_InterpretableLike:$lhs,
        Base2_InterpretableLike:$rhs);
    let results = (outs BoolLike:$result);

    let assemblyFormat = [{
        $predicate $lhs `,` $rhs attr-dict `:` type($lhs)
    }];

    let hasFolder = 1;

    code extraClassDeclaration = [{
    public:
        //===--------------------------------------------------------------===//
        // OpAsmOpInterface
        //===--------------------------------------------------------------===//

        void getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn)
        {
            setNameFn(
                getResult(),
                stringifyPartialOrderingPredicate(getPredicate()));
        }
    }];
}

class Base2_MinMaxOpBase<string mnemonic, list<Trait> traits = []>
        :  Base2_Op<
            mnemonic,
            traits
                # [
                    Pure,
                    OpAsmOpInterface,
                    SameOperandsAndResultType,
                    Commutative]
                # ElementwiseMappable.traits> {
    let arguments = (ins
        Base2_InterpretableLike:$lhs,
        Base2_InterpretableLike:$rhs);
    let results = (outs Base2_InterpretableLike:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` type($lhs)
    }];

    let hasFolder = 1;
}

def Base2_MinOp : Base2_MinMaxOpBase<"min"> {
    let summary = "Minimum value of two interpretable bit sequences";
    let description = [{
        The `min` operation obtains the smallest value of two interpretable bit
        sequences. This may differ from `cmp` and `select`, as interpretation
        of this operation may propagate special values.

        Syntax:

        ```
        min-op ::= `min` $lhs, $rhs `:` type($lhs)
        ```

        Example:

        ```
        %0 = base2.min %1, %2 : si16
        ```
    }];

    code extraClassDeclaration = [{
    public:
        //===--------------------------------------------------------------===//
        // OpAsmOpInterface
        //===--------------------------------------------------------------===//

        void getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn)
        {
            setNameFn(getResult(), "min");
        }
    }];
}

def Base2_MaxOp : Base2_MinMaxOpBase<"max"> {
    let summary = "Maximum value of two interpretable bit sequences";
    let description = [{
        The `max` operation obtains the largest value of two interpretable bit
        sequences. This may differ from `cmp` and `select`, as interpretation
        of this operation may propagate special values.

        Syntax:

        ```
        max-op ::= `max` $lhs, $rhs `:` type($lhs)
        ```

        Example:

        ```
        %0 = base2.max %1, %2 : si16
        ```
    }];

    code extraClassDeclaration = [{
    public:
        //===--------------------------------------------------------------===//
        // OpAsmOpInterface
        //===--------------------------------------------------------------===//

        void getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn)
        {
            setNameFn(getResult(), "max");
        }
    }];
}

//===----------------------------------------------------------------------===//
// Closed arithmetic operations
//===----------------------------------------------------------------------===//

// Template for closed arithmetic ops.
class Base2_ClosedArithmeticOpBase<string mnemonic, list<Trait> traits = []>
        : Base2_Op<
            mnemonic,
            traits
                # [
                    Pure,
                    OpAsmOpInterface,
                    Base2_ArithmeticOp,
                    Base2_RoundingOp,
                    SameOperandsAndResultType]
                # ElementwiseMappable.traits> {
    let arguments = (ins
        Base2_InterpretableLike:$lhs,
        Base2_InterpretableLike:$rhs,
        Base2_RoundingModeDefaultAttr:$roundingMode);
    let results = (outs Base2_InterpretableLike:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` ($roundingMode^)? type($result)
    }];

    let hasFolder = 1;
}

def Base2_AddOp : Base2_ClosedArithmeticOpBase<"add", [Commutative]> {
    let summary = "Sum of two interpretable bit sequences";
    let description = [{
        The `add` operation computes the sum of two interpretable bit sequences.
        If the value is not exactly representable, the specified rounding is
        applied.

        Syntax:

        ```
        add-op ::= `add` $lhs, $rhs `:` ($roundingMode)? type($lhs)
        ```

        Example:

        ```
        %0 = base2.add %1, %2 : nearest si16
        ```
    }];

    code extraClassDeclaration = [{
    public:
        //===--------------------------------------------------------------===//
        // OpAsmOpInterface
        //===--------------------------------------------------------------===//

        void getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn)
        {
            setNameFn(getResult(), "sum");
        }
    }];
}

def Base2_SubOp : Base2_ClosedArithmeticOpBase<"sub"> {
    let summary = "Difference of two interpretable bit sequences";
    let description = [{
        The `sub` operation computes the difference of two interpretable bit
        sequences. If the value is not exactly representable, the specified
        rounding is applied.

        Syntax:

        ```
        sub-op ::= `sub` $lhs, $rhs `:` ($roundingMode)? type($lhs)
        ```

        Example:

        ```
        %0 = base2.sub %1, %2 : nearest si16
        ```
    }];

    code extraClassDeclaration = [{
    public:
        //===--------------------------------------------------------------===//
        // OpAsmOpInterface
        //===--------------------------------------------------------------===//

        void getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn)
        {
            setNameFn(getResult(), "diff");
        }
    }];
}

def Base2_MulOp : Base2_ClosedArithmeticOpBase<"mul", [Commutative]> {
    let summary = "Product of two interpretable bit sequences";
    let description = [{
        The `mul` operation computes the product of two interpretable bit
        sequences. If the value is not exactly representable, the specified
        rounding is applied.

        Syntax:

        ```
        mul-op ::= `mul` $lhs, $rhs `:` ($roundingMode)? type($lhs)
        ```

        Example:

        ```
        %0 = base2.mul %1, %2 : nearest si16
        ```
    }];

    code extraClassDeclaration = [{
    public:
        //===--------------------------------------------------------------===//
        // OpAsmOpInterface
        //===--------------------------------------------------------------===//

        void getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn)
        {
            setNameFn(getResult(), "prod");
        }
    }];
}

def Base2_DivOp : Base2_ClosedArithmeticOpBase<"div"> {
    let summary = "Quotient of two interpretable bit sequences";
    let description = [{
        The `div` operation computes the quotient of two interpretable bit
        sequences. If the value is not exactly representable, the specified
        rounding is applied.

        Syntax:

        ```
        div-op ::= `div` $lhs, $rhs `:` ($roundingMode)? type($lhs)
        ```

        Example:

        ```
        %0 = base2.div %1, %2 : nearest si16
        ```
    }];

    code extraClassDeclaration = [{
    public:
        //===--------------------------------------------------------------===//
        // OpAsmOpInterface
        //===--------------------------------------------------------------===//

        void getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn)
        {
            setNameFn(getResult(), "quot");
        }
    }];
}

def Base2_ModOp
        : Base2_Op<
            "mod",
            [
                Pure,
                OpAsmOpInterface,
                Base2_ArithmeticOp,
                SameOperandsAndResultType]
                # ElementwiseMappable.traits> {
    let arguments = (ins
        Base2_InterpretableLike:$lhs,
        Base2_InterpretableLike:$rhs);
    let results = (outs Base2_InterpretableLike:$result);

    let summary = "Remainder of dividing two interpretable bit sequences";
    let description = [{
        The `mod` operation computes the remainder of dividing two interpretable
        bit sequences. If the value is not exactly representable, the specified
        rounding is applied.

        Syntax:

        ```
        mod-op ::= `mod` $lhs, $rhs `:` type($lhs)
        ```

        Example:

        ```
        %0 = base2.mod %1, %2 : si16
        ```
    }];

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` type($result)
    }];

    let hasFolder = 1;

    code extraClassDeclaration = [{
    public:
        //===--------------------------------------------------------------===//
        // OpAsmOpInterface
        //===--------------------------------------------------------------===//

        void getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn)
        {
            setNameFn(getResult(), "rem");
        }
    }];
}

//===----------------------------------------------------------------------===//
// Fixed-point operations
//===----------------------------------------------------------------------===//

include "base2-mlir/Dialect/Base2/IR/FixedOps.td"

//===----------------------------------------------------------------------===//
// Floating-point operations
//===----------------------------------------------------------------------===//

include "base2-mlir/Dialect/Base2/IR/FloatOps.td"

#endif
