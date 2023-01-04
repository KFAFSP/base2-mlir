//===- BitSequenceType.td ----------------------------------*- tablegen -*-===//
//
// This is the definitions file for the Base2 BitSequenceType interface.
//
//===----------------------------------------------------------------------===//

#ifndef BASE2_BITSEQUENCETYPE
#define BASE2_BITSEQUENCETYPE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// BitSequenceType
//===----------------------------------------------------------------------===//

def Base2_BitSequenceType : TypeInterface<"BitSequenceType"> {
    let cppNamespace = "::mlir::base2";

    let description = [{
        Interface for a type that is represented by a bit sequence with a fixed
        layout.
    }];

    let methods = [
        InterfaceMethod<
            /*desc=*/"Gets the number of bits in this bit sequence.",
            /*retTy=*/"bit_width_t",
            /*methodName=*/"getBitWidth",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/"">];
}

//===----------------------------------------------------------------------===//
// BitSequenceLike
//===----------------------------------------------------------------------===//

def Base2_BitSequenceLike
        : TypeConstraint<
            /*predicate=*/CPred<"$_self.isa<::mlir::base2::BitSequenceLikeType>()">,
            /*summary=*/"Type or container of BitSequenceType",
            /*cppClassName=*/"::mlir::base2::BitSequenceLikeType">;

#endif
