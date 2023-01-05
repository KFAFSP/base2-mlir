//===- Ops.td - SoftFloat dialect ops ----------------------*- tablegen -*-===//
//
// This is the definitions file for the SoftFloat dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef SOFTFLOAT_OPS
#define SOFTFLOAT_OPS

include "base2-mlir/Dialect/SoftFloat/IR/Types.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class SoftFloat_ArithmeticOp<string mnemonic, list<Trait> traits = []>
    : SoftFloat_Op<mnemonic, traits # [SameOperandsAndResultShape]> {
        let arguments = (ins SoftFloat_SFloatType:$lhs, SoftFloat_SFloatType:$rhs,
                             I8:$exp_bits, I8:$frac_bits, I32:$exp_bias,
                             I1:$rounding, I1:$nan, I1:$one, I1:$subnorm,
                             I8:$sign);
        let results = (outs SoftFloat_SFloatType:$result);
        let assemblyFormat = "$lhs `,` $rhs `(` $exp_bits `,` $frac_bits `,` $exp_bias `,` $rounding `,` $nan `,` $one `,` $subnorm `,` $sign `)` attr-dict `:` type($result)";
}

class SoftFloat_ComparisonOp<string mnemonic, list<Trait> traits = []>
    : SoftFloat_Op<mnemonic, traits # [SameOperandsAndResultShape]> {
        let arguments = (ins SoftFloat_SFloatType:$lhs, SoftFloat_SFloatType:$rhs,
                             I8:$exp_bits, I8:$frac_bits, I32:$exp_bias,
                             I1:$rounding, I1:$nan, I1:$one, I1:$subnorm,
                             I8:$sign);
        let results = (outs I1:$result);
        let assemblyFormat = "$lhs `,` $rhs `(` $exp_bits `,` $frac_bits `,` $exp_bias `,` $rounding `,` $nan `,` $one `,` $subnorm `,` $sign `)` attr-dict `:` type($result)";
    }

/***
CastOp
***/

def SoftFloat_CastOp : SoftFloat_Op<"cast", [SameOperandsAndResultShape]> {
    let summary = "cast a f64 number into sfloat type";
    let description = [{
        The `sfloat.cast` operation creates a 64-bit sfloat number from
        a standard IEEE754 double precision number and the output semantics, which will be lowering to
        a cast function in softfloat lib i.e. __float_cast();

        Example: (hopefully)
        ```
        %1 = softfloat.constant ... : sfloat
        ```
    }];

    let arguments = (ins SoftFloat_SFloatType:$in,
                         I8:$in_exp_bits, I8:$in_frac_bits, I32:$in_exp_bias,
                         I1:$in_has_rounding, I1:$in_has_nan, I1:$in_has_one, I1:$in_has_subnorm,
                         I8:$in_sign,
                         I8:$out_exp_bits, I8:$out_frac_bits, I32:$out_exp_bias,
                         I1:$out_has_rounding, I1:$out_has_nan, I1:$out_has_one, I1:$out_has_subnorm,
                         I8:$out_sign);
    let results = (outs SoftFloat_SFloatType:$out);

    let assemblyFormat = "$in `(` $in_exp_bits `,` $in_frac_bits `,` $in_exp_bias `,` $in_has_rounding `,` $in_has_nan `,` $in_has_one `,` $in_has_subnorm `,` $in_sign `,` $out_exp_bits `,` $out_frac_bits `,` $out_exp_bias `,` $out_has_rounding `,` $out_has_nan `,` $out_has_one `,` $out_has_subnorm `,` $out_sign `)` attr-dict `:` type($out)";
}

def SoftFloat_CastFloatOp : SoftFloat_Op<"castfloat", [SameOperandsAndResultShape]> {
    let summary = "cast a normal f64 number into sfloat type";
    let description = [{
        The `softflaot.floatcast` operation creates a 64-bit sfloat number from
        a standard float 64-bit number into a bit-wise representation, and then into
        a 64-bit sfloat number according to the semantics procided. 
    }];

    let arguments = (ins F64:$in,
                     I8:$exp_bits, I8:$frac_bits, I32:$exp_bias,
                     I1:$has_rounding, I1:$has_nan, I1:$has_one, I1:$has_subnorm,
                     I8:$sign);
    let results = (outs SoftFloat_SFloatType:$out);

    let assemblyFormat = "$in `(` $exp_bits `,` $frac_bits `,` $exp_bias `,` $has_rounding `,` $has_nan `,` $has_one `,` $has_subnorm `,` $sign `)` attr-dict `:` type($out)";
}

def SoftFloat_CastToFloatOp : SoftFloat_Op<"casttofloat", [SameOperandsAndResultShape]> {
    let summary = "cast a 64-bit sfloat number into f64";
    let description = [{
        The `softflaot.casttofloat` operation does the opposite thing
        as `softfloat.castfloat`. 
    }];

    let arguments = (ins SoftFloat_SFloatType:$in,
                     I8:$exp_bits, I8:$frac_bits, I32:$exp_bias,
                     I1:$has_rounding, I1:$has_nan, I1:$has_one, I1:$has_subnorm,
                     I8:$sign);
    let results = (outs F64:$out);

    let assemblyFormat = "$in `(` $exp_bits `,` $frac_bits `,` $exp_bias `,` $has_rounding `,` $has_nan `,` $has_one `,` $has_subnorm `,` $sign `)` attr-dict `:` type($out)";
}

/***
Arithmetic operations between two 64-bit SFloat number.
***/

def SoftFloat_AddOp : SoftFloat_ArithmeticOp<"add", []> {
    let summary = "addition operation";
    let description = [{
        The "add operation between two arbitrary floating-point
        number following IEEE754 Standard using SoftFloat Library.
        The numbers are up to 64bit double precision."
    }];
}

def SoftFloat_SubOp : SoftFloat_ArithmeticOp<"sub", []> {
    let summary = "subtraction operation";
}

def SoftFloat_MulOp : SoftFloat_ArithmeticOp<"mul", []> {
    let summary = "multiplication operation";
}

def SoftFloat_DivSRTOp : SoftFloat_ArithmeticOp<"divsrt", []> {
    let summary = "division operation using SRT4 algorithm";
}

def SoftFloat_DivGOp : SoftFloat_ArithmeticOp<"divg", []> {
    let summary = "general division operation";
}

/***
Comparison operations between two 64-bit SFloat number
***/

def SoftFloat_EQOp : SoftFloat_ComparisonOp<"eq", []> {
    let summary = "equation comparison";
    let description = [{
        This operation returns True when two operands are
        equal, otherwise 0.
    }];
}

def SoftFloat_LEOp : SoftFloat_ComparisonOp<"le", []> {
    let summary = "less than or euqal";
}

def SoftFloat_LTOp : SoftFloat_ComparisonOp<"lt", []> {
    let summary = "less than";
}

def SoftFloat_GEOp : SoftFloat_ComparisonOp<"ge", []> {
    let summary = "greater than or euqal";
}

def SoftFloat_GTOp : SoftFloat_ComparisonOp<"gt", []> {
    let summary = "greater than";
}

def SoftFloat_LTGTOp : SoftFloat_ComparisonOp<"ltgt", []> {
    let summary = "less than or greater than";
}

def SoftFloat_NaNOp : SoftFloat_Op<"nan", [SameOperandsAndResultShape]> {
    let summary = "is a signaling NaN";
    let description = [{
        This operation returns True when a number is a signaling
        NaN number, otherwise 0.
    }];

    let arguments = (ins SoftFloat_SFloatType:$in,
                         I8:$exp_bits, I8:$frac_bits, I32:$exp_bias,
                         I1:$rounding, I1:$nan, I1:$one, I1:$subnorm,
                         I8:$sign);
    let results = (outs I1:$result);
    let assemblyFormat = "$in `(` $exp_bits `,` $frac_bits `,` $exp_bias `,` $rounding `,` $nan `,` $one `,` $subnorm `,` $sign `)` attr-dict `:` type($result)";
}

#endif
