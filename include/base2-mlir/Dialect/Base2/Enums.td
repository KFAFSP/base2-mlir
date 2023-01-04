//===- Enums.td --------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the Base2 enumerations.
//
//===----------------------------------------------------------------------===//

#ifndef BASE2_ENUMS
#define BASE2_ENUMS

include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Signedness
//===----------------------------------------------------------------------===//

// NOTE: Unfortunately, TableGen cannot produce an EnumAttr for a C++ enum that
//       already exists, and will instead always emit a declaration for one.
//       This means we cannot use the IntegerType::SignednessSemantics enum as
//       the backing enum, even though these are equivalent.

def Base2_SignednessAttr : I64EnumAttr<
        "Signedness",
        "Signedness semantics of a number type",
        [
            I64EnumAttrCase<"Signless", 0b00, "signless">,
            I64EnumAttrCase<"Signed",   0b01, "signed">,
            I64EnumAttrCase<"Unsigned", 0b11, "unsigned">]> {
    let cppNamespace = "::mlir::base2";
}

def Base2_SignednessDefaultAttr
        : DefaultValuedAttr<Base2_SignednessAttr, "Signedness::Signless">;

//===----------------------------------------------------------------------===//
// PartialOrderingPredicate
//===----------------------------------------------------------------------===//

// NOTE: This is equivalent to the "Arith_CmpFPredicateAttr" declared in
//       "mlir/Dialect/Arith/IR/ArithBase.td". However, we use it for all
//       comparisons, including fixed-point, and make explicit use of its bit
//       flags. Thus, we declare our own.

// NOTE: The bit structure is 0b_ulge, where each bit indicates which relation
//       is matched by the predicate.

def Base2_PartialOrderingPredicateAttr : I64EnumAttr<
        "PartialOrderingPredicate",
        "Predicate to test for during comparison of two numbers",
        [
            I64EnumAttrCase<"Falsum",                    0b0000, "false">,
            I64EnumAttrCase<"OrderedAndEqual",           0b0001, "oeq">,
            I64EnumAttrCase<"OrderedAndGreater",         0b0010, "ogt">,
            I64EnumAttrCase<"OrderedAndGreaterOrEqual",  0b0011, "oge">,
            I64EnumAttrCase<"OrderedAndLess",            0b0100, "olt">,
            I64EnumAttrCase<"OrderedAndLessOrEqual",     0b0101, "ole">,
            I64EnumAttrCase<"OrderedAndUnequal",         0b0110, "one">,
            I64EnumAttrCase<"Ordered",                   0b0111, "ord">,
            I64EnumAttrCase<"Unordered",                 0b1000, "uno">,
            I64EnumAttrCase<"UnorderedOrEqual",          0b1001, "ueq">,
            I64EnumAttrCase<"UnorderedOrGreater",        0b1010, "ugt">,
            I64EnumAttrCase<"UnorderedOrGreaterOrEqual", 0b1011, "uge">,
            I64EnumAttrCase<"UnorderedOrLess",           0b1100, "ult">,
            I64EnumAttrCase<"UnorderedOrLessOrEqual",    0b1101, "ule">,
            I64EnumAttrCase<"UnorderedOrUnequal",        0b1110, "une">,
            I64EnumAttrCase<"Verum",                     0b1111, "true">]> {
    let cppNamespace = "::mlir::base2";
}

//===----------------------------------------------------------------------===//
// RoundingMode
//===----------------------------------------------------------------------===//

// NOTE: The bit structure is 0b_LRCDT_udlr, where each bit indicates a property
//       of the rounding guarantee.
//
//       The upper LRCDT bits apply to floating-point numbers:
//
//       - LRCDT = +inf, -inf, converge, directed, ties
//       - 00000 = Towards 0
//       - 00001 = Nearest, ties to even
//       - 10010 = Towards +inf
//       - 01010 = Towards -inf
//       - 00010 = Towards 0
//       - 11010 = Away from 0
//       - 00100 = Convergent rounding
//
//       The lower udlr bits apply to fixed-point numbers:
//
//       - udlr = saturate max, saturate min, mod max, mod min
//       - 0011 = Modulo on over- and underflow
//       - 1100 = Saturate on over- and underflow
//       - 0100 = Saturate on underflow
//       - 1000 = Saturate on overflow
//       - 1001 = Modulo on overflow, saturate on underflow
//       - 0110 = Saturate on overflow, modulo on underflow
//       - 0000 = Undefined behavior

def Base2_RoundingModeAttr : I64EnumAttr<
        "RoundingMode",
        "Rounding behavior",
        [
            I64EnumAttrCase<"None",         0b000000011, "none">,
            I64EnumAttrCase<"Nearest",      0b000011100, "nearest">,
            I64EnumAttrCase<"RoundUp",      0b100100100, "round_up">,
            I64EnumAttrCase<"RoundDown",    0b010101000, "round_down">,
            I64EnumAttrCase<"TowardsZero",  0b000100101, "towards_zero">,
            I64EnumAttrCase<"AwayFromZero", 0b110101010, "away_from_zero">,
            I64EnumAttrCase<"Converge",     0b001000000, "converge">
        ]> {
    let cppNamespace = "::mlir::base2";
}

def Base2_RoundingModeDefaultAttr
        : DefaultValuedAttr<Base2_RoundingModeAttr, "RoundingMode::None">;

#endif
