//===- LogicOp.td ------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the Bit LogicOp interface.
//
//===----------------------------------------------------------------------===//

#ifndef BIT_LOGICOP
#define BIT_LOGICOP

include "mlir/IR/OpBase.td"

def Bit_LogicOp : OpInterface<"LogicOp"> {
    let cppNamespace = "::mlir::bit";

    let description = [{
        Interface for an operation that implements a bitwise logical operation.
    }];

    code verify = [{
        static_assert(
            ConcreteOp::template hasTrait<::mlir::OpTrait::SameOperandsAndResultType>(),
            "`LogicOp` is only applicable to `SameOperandsAndResultType` ops."
        );
        static_assert(
            ConcreteOp::template hasTrait<::mlir::OpTrait::OneResult>(),
            "`LogicOp` is only applicable to `OneResult` ops."
        );
        static_assert(
            ConcreteOp::template hasTrait<::mlir::OpTrait::Elementwise>()
            && ConcreteOp::template hasTrait<::mlir::OpTrait::Scalarizable>()
            && ConcreteOp::template hasTrait<::mlir::OpTrait::Vectorizable>()
            && ConcreteOp::template hasTrait<::mlir::OpTrait::Tensorizable>(),
            "`LogicOp` is only applicable to `ElementwiseMappable` ops."
        );

        return ::mlir::success();
    }];
}

#endif
