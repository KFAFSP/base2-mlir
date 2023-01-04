//===- RoundingOp.td ---------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the Base2 RoundingOp interface.
//
//===----------------------------------------------------------------------===//

#ifndef BASE2_ROUNDINGOP
#define BASE2_ROUNDINGOP

include "mlir/IR/OpBase.td"

def Base2_RoundingOp : OpInterface<"RoundingOp"> {
    let cppNamespace = "::mlir::base2";

    let description = [{
        Interface for an operation that may perform rounding.
    }];

    let methods = [
        InterfaceMethod<
            /*desc=*/[{
                Gets the RoundingMode applied when an exact result can not be
                returned.
            }],
            /*retTy=*/"RoundingMode",
            /*methodName=*/"getRoundingMode",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/[{
                return rounding_op_interface_defaults::getRoundingMode($_op);
            }]>,
        InterfaceMethod<
            /*desc=*/[{
                Marks this operation as known-exact, meaning that no rounding
                will ever have to be applied at runtime-time.
            }],
            /*retTy=*/"void",
            /*methodName=*/"markAsKnownExact",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/[{
                rounding_op_interface_defaults::markAsKnownExact($_op);
            }]>];

    code extraClassDeclaration = [{
    public:
        /// Gets the default name of the RoundingMode attribute.
        static StringRef getRoundingModeAttrName() { return "roundingMode"; }
    }];
}

#endif
