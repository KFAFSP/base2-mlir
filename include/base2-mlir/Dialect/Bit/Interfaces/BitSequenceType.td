//===- BitSequenceType.td ----------------------------------*- tablegen -*-===//
//
// This is the definitions file for the Bit BitSequenceType interface.
//
//===----------------------------------------------------------------------===//

#ifndef BIT_BITSEQUENCETYPE
#define BIT_BITSEQUENCETYPE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// BitSequenceType
//===----------------------------------------------------------------------===//

def Bit_BitSequenceType : TypeInterface<"BitSequenceType"> {
    let cppNamespace = "::mlir::bit";

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

def Bit_BitSequenceLike
        : Type<
            /*predicate=*/CPred<"$_self.isa<::mlir::bit::BitSequenceLikeType>()">,
            /*summary=*/"Type or container of BitSequenceType",
            /*cppClassName=*/"::mlir::bit::BitSequenceLikeType">;

#endif
