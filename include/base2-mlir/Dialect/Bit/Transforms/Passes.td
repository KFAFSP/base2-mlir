//===- Passes.td -------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the Bit passes.
//
//===----------------------------------------------------------------------===//

#ifndef BIT_PASSES
#define BIT_PASSES

include "mlir/Pass/PassBase.td"

def LowerBitwiseLogic : Pass<"lower-bitwise-logic", "ModuleOp"> {
    let summary = "Lowers bitwise logic operations to signless integers";
    let description = [{
        This pass replaces all bitwise logic operations with operations that
        operate on signless integers.

        The `bit.and`, `bit.or` and `bit.xor` operations have their operands
        converted using `bit.cast` operations.
    }];

    let constructor = "mlir::bit::createLowerBitwiseLogicPass()";
}

#endif
