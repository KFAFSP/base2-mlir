//===- ArithmeticOp.td -------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the Base2 ArithmeticOp interface.
//
//===----------------------------------------------------------------------===//

#ifndef BASE2_ARITHMETICOP
#define BASE2_ARITHMETICOP

include "mlir/IR/OpBase.td"

def Base2_ArithmeticOp : OpInterface<"ArithmeticOp"> {
    let cppNamespace = "::mlir::base2";

    let description = [{
        Interface for an operation that implements a closed arithmetic
        operation.
    }];

    code verify = [{
        static_assert(
            ConcreteOp::template hasTrait<::mlir::OpTrait::SameOperandsAndResultType>(),
            "`ArithmeticOp` is only applicable to `SameOperandsAndResultType` ops."
        );
        static_assert(
            ConcreteOp::template hasTrait<::mlir::OpTrait::Elementwise>()
            && ConcreteOp::template hasTrait<::mlir::OpTrait::Scalarizable>()
            && ConcreteOp::template hasTrait<::mlir::OpTrait::Vectorizable>()
            && ConcreteOp::template hasTrait<::mlir::OpTrait::Tensorizable>(),
            "`ArithmeticOp` is only applicable to `ElementwiseMappable` ops."
        );

        return ::mlir::success();
    }];
}

#endif
