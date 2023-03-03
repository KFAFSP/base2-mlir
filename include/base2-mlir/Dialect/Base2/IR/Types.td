//===- Types.td --------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the Base2 dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef BASE2_TYPES
#define BASE2_TYPES

include "base2-mlir/Dialect/Base2/IR/Attributes.td"
include "base2-mlir/Dialect/Base2/Interfaces/FixedPointSemantics.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//===----------------------------------------------------------------------===//
// FixedPointType
//===----------------------------------------------------------------------===//

def Base2_FixedPointType : Base2_Type<
        "FixedPoint",
        [
            Base2_BitSequenceType,
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
        fixed-type ::= `fixed` `<` (signedness)? $bitWidth
                                   (`,` $exponent)? `>`
        signedness ::= `signed` | `unsigned`
        ```

        An abbreviated syntax is also supported, which mimics the builtin
        integer type literals:

        ```
        fixed-sema      ::= (signedness-spec)? `i` [0-9]+ (`_` (m)?[0-9]+)?
        signedness-spec ::= `s` | `u`
        ```

        Example:

        ```
        !base2.fixed_point<11,3>           = !base2.fixed_point<signless 11,3>
                                           = !base2.i11_m3
        !base2.fixed_point<unsigned 8,-5>  = !base2.fixed_point<unsigned 8,-5>
                                           = !base2.ui8_5
        !base2.fixed_point<signed 12>      = !base2.si12
        ```
    }];

    let parameters = (ins
        "IntegerType":$integerType,
        "exponent_t":$exponent);

    let builders = [
        TypeBuilderWithInferredContext<(ins
            "IntegerType":$integerType,
            CArg<"exponent_t", "0">:$exponent),
        [{
            return $_get(integerType.getContext(), integerType, exponent);
        }]>,
        TypeBuilder<(ins
                "Signedness":$signedness,
                "bit_width_t":$bitWidth,
                CArg<"exponent_t", "0">:$exponent),
        [{
            return $_get(
                $_ctxt,
                IntegerType::get(
                    $_ctxt,
                    bitWidth,
                    getBuiltinSignedness(signedness)),
                exponent);
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
            bit_width_t integerBits);
    }];
}

#endif
