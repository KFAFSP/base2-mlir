//===- Types.td --------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the Base2 dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef BASE2_TYPES
#define BASE2_TYPES

include "base2-mlir/Dialect/Base2/IR/Attributes.td"
include "base2-mlir/Dialect/Base2/Interfaces/FixedPointSemantics.td"
include "base2-mlir/Dialect/Base2/Interfaces/IEEE754Semantics.td"
include "base2-mlir/Dialect/Bit/Interfaces/BitSequenceType.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//===----------------------------------------------------------------------===//
// FixedPointType
//===----------------------------------------------------------------------===//

def Base2_FixedPointType : Base2_Type<
        "FixedPoint",
        [
            Bit_BitSequenceType,
            Base2_FixedPointSemantics,
            DeclareTypeInterfaceMethods<
                Base2_InterpretableType,
                [
                    "valueCast",
                    "cmp",
                    "add",
                    "sub",
                    "mul",
                    "div",
                    "mod",
                    "getFacts"]>]> {
    let mnemonic = "fixed_point";
    let summary = "Fixed-point number type";
    let description = [{
        The `fixed_point` type stores a fixed-point rational number using the
        specified number of integer and fractional bits. It may also have a
        signedness, in which case it uses the two's complement representation.

        Syntax:

        ```
        fixed-type ::= `fixed` `<` (signedness)? $integerBits
                                   (`,` $fractionalBits)? `>`
        signedness ::= `signed` | `unsigned`
        ```

        An abbreviated syntax is also supported, which mimics the builtin
        integer type literals:

        ```
        fixed-sema      ::= (signedness-spec)? `i` [0-9]+ (`_` [0-9]+)?
        signedness-spec ::= `s` | `u`
        ```

        Example:

        ```
        !base2.fixed_point<11,3>         = !base2.fixed_point<signless 11,3>
                                         = !base2.i11_3
        !base2.fixed_point<signed 12>    = !base2.si12
        !base2.fixed_point<unsigned 0,4> = !base2.ui0_4
        ```
    }];

    let parameters = (ins
        "IntegerType":$integerType,
        "bit::bit_width_t":$fractionalBits);

    let builders = [
        TypeBuilderWithInferredContext<(ins
            "IntegerType":$integerType,
            CArg<"bit::bit_width_t", "0">:$fractionalBits),
        [{
            return $_get(integerType.getContext(), integerType, fractionalBits);
        }]>,
        TypeBuilder<(ins
                "Signedness":$signedness,
                "bit::bit_width_t":$integerBits,
                CArg<"bit::bit_width_t", "0">:$fractionalBits),
        [{
            return $_get(
                $_ctxt,
                IntegerType::get(
                    $_ctxt,
                    integerBits + fractionalBits,
                    getBuiltinSignedness(signedness)),
                fractionalBits);
        }]>];
    let skipDefaultBuilders = 1;

    let hasCustomAssemblyFormat = 1;

    let genVerifyDecl = 1;

    code extraClassDeclaration = [{
    public:
        /// Verifies type arguments used during custom parsing.
        static LogicalResult verify(
            function_ref<InFlightDiagnostic()> emitError,
            Signedness signedness,
            bit::bit_width_t integerBits,
            bit::bit_width_t fractionalBits);

        //===--------------------------------------------------------------===//
        // BitSequenceType
        //===--------------------------------------------------------------===//

        [[nodiscard]] bit::bit_width_t getBitWidth() const
        {
            return getIntegerType().getWidth();
        }
    }];
}

//===----------------------------------------------------------------------===//
// IEEE754Type
//===----------------------------------------------------------------------===//

def Base2_IEEE754Type : Base2_Type<
        "IEEE754",
        [
            Bit_BitSequenceType,
            Base2_IEEE754Semantics,
            DeclareTypeInterfaceMethods<Base2_InterpretableType, []>]> {
    let mnemonic = "ieee754";
    let summary = "IEEE754 floating-point number type";
    let description = [{
        The `ieee754` type defines an IEEE-754 floating-point number format
        using the bit width of the mantissa and the exponent, as well as the
        exponent bias value.

        All `ieee754` types adopt:

        - The sign-magnitude representation (sign bit).
        - The hidden one convention (denormalized numbers).
        - Can represent NaN/Inf (special max biased exponent value).

        Syntax:

        ```
        ieee754-type ::= `ieee754` `<` $precision `,` $exponentBits
                                       ( `,` $bias )? `>`
        ```

        If no bias is specified, the half-point between the minimum and maximum
        exponent within the given exponent bits is chosen. This behavior mirrors
        the standard types:

        - `ieee754<23, 8, 127>`   = `ieee754<23, 8>`  ~ `f32`
        - `ieee754<52, 11, 1023>` = `ieee754<52, 11>` ~ `f64`

        An abbreviated syntax is also supported, which mimics the builtin
        float type literals:

        ```
        ieee754-sema ::= `f` [0-9]+ `_` [0-9]+ (bias-spec)?
        bias-spec    ::= (`m`)? [0-9]+
        ```

        Example:

        ```
        !base2.ieee754<23, 8>  = !base2.f23_8  ~ f32
        !base2.ieee754<52, 11> = !base2.f52_11 ~ f64
        ```
    }];

    let parameters = (ins
        "bit::bit_width_t":$precision,
        "bit::bit_width_t":$exponentBits,
        "exponent_t":$bias);

    let builders = [
        TypeBuilder<(ins
            "bit::bit_width_t":$precision,
            "exponent_t":$maxExponent,
            "exponent_t":$minExponent),
        [{
            const auto [exponentBits, bias] =
                getExponentParams(maxExponent, minExponent);
            return $_get($_ctxt, precision, exponentBits, bias);
        }]>,
        TypeBuilder<(ins "bit::bit_width_t":$precision, "exponent_t":$maxExponent),
        [{
            return $_get(
                $_ctxt,
                precision,
                maxExponent,
                exponent_t(1) - maxExponent);
        }]>];

    let hasCustomAssemblyFormat = 1;

    let genVerifyDecl = 1;

    code extraClassDeclaration = [{
    }];
}

#endif
