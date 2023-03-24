//===- Ops.td ----------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the Bit dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef BIT_OPS
#define BIT_OPS

include "base2-mlir/Dialect/Bit/IR/Types.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

def Bit_ConstantOp : Bit_Op<
        "constant",
        [
            ConstantLike,
            Pure,
            OpAsmOpInterface,
            DeclareOpInterfaceMethods<
                InferTypeOpInterface,
                ["inferReturnTypes"]>]> {
    let summary = "Declares a constant value using a `BitSequenceLikeAttr`";
    let description = [{
        The `constant` operation produces a value or a container of values of a
        `BitSequenceType` using a `BitSequenceLikeAttr`.

        Syntax:

        ```
        constant-op ::= `constant` $value
        ```

        Examples:

        ```
        // The following are all equivalent:
        %0 = bit.constant 1.342000e+01 : f32
        %1 = bit.constant "0b_0_10000010_10101101011100001010010" : f32
        %2 = bit.constant "0x4156b852" : f32

        // The following are all equivalent:
        %3 = bit.constant #bit.dense_bits<tensor<3xi16> = dense<[1, 2, 3]>>
        %4 = bit.constant #bit.dense_bits<tensor<3xi16> = dense<"0x010002000300">>
        ```
    }];

    let arguments = (ins Bit_BitSequenceLikeAttr:$value);
    let results = (outs Bit_BitSequenceLike:$result);

    let assemblyFormat = "attr-dict $value";

    let hasFolder = 1;
    let hasCanonicalizer = 1;

    code extraClassDeclaration = [{
    public:
        //===--------------------------------------------------------------===//
        // OpAsmOpInterface
        //===--------------------------------------------------------------===//

        void getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn)
        {
            setNameFn(getResult(), "cst");
        }
    }];
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

def Bit_CastOp : Bit_Op<
        "cast",
        [
            Pure,
            SameOperandsAndResultShape,
            DeclareOpInterfaceMethods<CastOpInterface>]> {
    let summary = "Bit casting for bit sequences";
    let description = [{
        The `cast` operation reinterprets a sequence of bits as a different
        type, without changing the bits. This may not preserve the value.

        Both the input and output operand must be bit sequence types or
        containers thereof, and they must have the same sequence length.

        Syntax:

        ```
        cast-op ::= `cast` $in `:` type($in) `to` type($out)
        ```

        Example:

        ```
        %0 = bit.cast %flt : f32 to i32
        ```
    }];

    let arguments = (ins Bit_BitSequenceLike:$in);
    let results = (outs Bit_BitSequenceLike:$out);

    let assemblyFormat = "$in attr-dict `:` type($in) `to` type($out)";

    let hasFolder = 1;
    let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// CmpOp
//===----------------------------------------------------------------------===//

def Bit_CmpOp : Bit_Op<
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
    let summary = "Bitwise comparison for bit sequences";
    let description = [{
        The `cmp` operation compares the two operands according to the specified
        predicate and returns a boolean or container of booleans indicating the
        result. The operands must have the same bit width.

        Syntax:

        ```
        cmp-op ::= `cmp` $predicate $lhs `,` $rhs `:` type(lhs)
        ```

        Example:

        ```
        %0 = bit.cmp eq %1, %2 : si16
        ```
    }];

    let arguments = (ins
        Bit_EqualityPredicateAttr:$predicate,
        Bit_BitSequenceLike:$lhs,
        Bit_BitSequenceLike:$rhs);
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
                stringifyEqualityPredicate(getPredicate()));
        }
    }];
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

def Bit_SelectOp : Bit_Op<
        "select",
        [
            Pure,
            AllTypesMatch<["trueValue", "falseValue", "result"]>,
            TypesMatchWith<
                "condition is i1 of result shape",
                "result", "condition", "::getI1SameShape($_self)">]> {
    let summary = "Ternary operator for bit sequences";
    let description = [{
        The `select` operation builds a result by examining a condition to
        selecting values from a binary alternative. If the condition is `i1`,
        the whole operand is selected. If the condition is a container of `i1`,
        the operand shapes must match, and an element is selected per index.

        Syntax:

        ```
        select-op ::= `select` $condition `,` $trueValue `,` $falseValue
                        `:` [ type($condition) `,` ] type($result)
        ```

        Example:

        ```
        %x = bit.select %cond, %t, %f : tensor<3xi32>
        %y = bit.select %cond2, %t, %f : tensor<3xi1>, tensor<3xi32>
        ```
    }];

    let arguments = (ins
        BoolLike:$condition,
        Bit_BitSequenceLike:$trueValue,
        Bit_BitSequenceLike:$falseValue);
    let results = (outs Bit_BitSequenceLike:$result);

    let hasCustomAssemblyFormat = 1;

    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Logic operations
//===----------------------------------------------------------------------===//

// Template for logic operations.
class Bit_LogicOpBase<string mnemonic, list<Trait> traits = []>
        : Bit_Op<
            mnemonic,
            traits
                # [
                    Pure,
                    OpAsmOpInterface,
                    Bit_LogicOp,
                    SameOperandsAndResultType]
                # ElementwiseMappable.traits> {
    let arguments = (ins
        Bit_BitSequenceLike:$lhs,
        Bit_BitSequenceLike:$rhs);
    let results = (outs Bit_BitSequenceLike:$result);

    let assemblyFormat = [{ operands attr-dict `:` type($result) }];

    let hasFolder = 1;
}

def Bit_AndOp : Bit_LogicOpBase<"and", [Commutative, Idempotent]> {
    let summary = "Bitwise logical and";
    let description = [{
        The `and` operation computes the bitwise logical and of two bit
        sequences or containers of bit sequences.

        Syntax:

        ```
        and-op ::= `and` $lhs `,` $rhs `:` type(result)
        ```

        Example:

        ```
        %0 = bit.and %1, %2 : i16
        ```
    }];
}

def Bit_OrOp : Bit_LogicOpBase<"or", [Commutative, Idempotent]> {
    let summary = "Bitwise logical or";
    let description = [{
        The `or` operation computes the bitwise logical or of two bit sequences
        or containers of bit sequences.

        Syntax:

        ```
        or-op ::= `or` $lhs `,` $rhs `:` type(result)
        ```

        Example:

        ```
        %0 = bit.or %1, %2 : i16
        ```
    }];
}

def Bit_XorOp : Bit_LogicOpBase<"xor", [Commutative]> {
    let summary = "Bitwise logical exclusive or";
    let description = [{
        The `xor` operation computes the bitwise logical exclusive or of two
        bit sequences or containers of bit sequences.

        Syntax:

        ```
        xor-op ::= `xor` $lhs `,` $rhs `:` type(result)
        ```

        Example:

        ```
        %0 = bit.xor %1, %2 : i16
        ```
    }];
}

#endif
