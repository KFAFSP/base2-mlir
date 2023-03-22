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

#endif
