//===- Enums.td --------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the Bit enumerations.
//
//===----------------------------------------------------------------------===//

#ifndef BIT_ENUMS
#define BIT_ENUMS

include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// EqualityPredicate
//===----------------------------------------------------------------------===//

// NOTE: The bit structure is 0b_ne, where each bit indicates which relation is
//       matched by the predicate.

def Bit_EqualityPredicateAttr : I64EnumAttr<
        "EqualityPredicate",
        "Predicate to test for during comparison of two bit sequences",
        [
            I64EnumAttrCase<"Falsum",  0b00, "false">,
            I64EnumAttrCase<"Equal",   0b01, "eq">,
            I64EnumAttrCase<"Unequal", 0b10, "ne">,
            I64EnumAttrCase<"Verum",   0b11, "true">]> {
    let cppNamespace = "::mlir::bit";
}

#endif
