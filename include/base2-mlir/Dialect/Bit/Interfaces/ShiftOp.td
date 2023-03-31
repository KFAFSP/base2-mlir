//===- ShiftOp.td ------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the Bit ShiftOp interface.
//
//===----------------------------------------------------------------------===//

#ifndef BIT_SHIFTOP
#define BIT_SHIFTOP

include "mlir/IR/OpBase.td"

def Bit_ShiftOp : OpInterface<"ShiftOp"> {
    let cppNamespace = "::mlir::bit";

    let description = [{
        Interface for an operation that implements a bit shift operation.
    }];

    let methods = [
        InterfaceMethod<
            /*desc=*/[{
                Gets the value with the bits to be shifted.
            }],
            /*retTy=*/"TypedValue<BitSequenceLikeType>",
            /*methodName=*/"getValue",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/"">,
        InterfaceMethod<
            /*desc=*/[{
                Gets the value with the funnel bits.
            }],
            /*retTy=*/"TypedValue<BitSequenceLikeType>",
            /*methodName=*/"getFunnel",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/"">,
        InterfaceMethod<
            /*desc=*/[{
                Gets the value with the shift amount.
            }],
            /*retTy=*/"TypedValue<IndexType>",
            /*methodName=*/"getAmount",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/"">,
        InterfaceMethod<
            /*desc=*/[{
                Gets the result value.
            }],
            /*retTy=*/"TypedValue<BitSequenceLikeType>",
            /*methodName=*/"getResult",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/"">,
        InterfaceMethod<
            /*desc=*/[{
                Gets the ShiftDirection.
            }],
            /*retTy=*/"ShiftDirection",
            /*methodName=*/"getDirection",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/"">];


    code verify = [{
        static_assert(
            ConcreteOp::template hasTrait<::mlir::OpTrait::OneResult>(),
            "`ShiftOp` is only applicable to `OneResult` ops."
        );
        static_assert(
            ConcreteOp::template hasTrait<::mlir::OpTrait::Elementwise>()
            && ConcreteOp::template hasTrait<::mlir::OpTrait::Scalarizable>()
            && ConcreteOp::template hasTrait<::mlir::OpTrait::Vectorizable>()
            && ConcreteOp::template hasTrait<::mlir::OpTrait::Tensorizable>(),
            "`ShiftOp` is only applicable to `ElementwiseMappable` ops."
        );

        return ::mlir::success();
    }];
}

#endif
