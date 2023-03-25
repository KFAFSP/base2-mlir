//===- Passes.td -------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the Bit passes.
//
//===----------------------------------------------------------------------===//

#ifndef BIT_PASSES
#define BIT_PASSES

include "mlir/Pass/PassBase.td"

def LowerBitwise : Pass<"lower-bitwise", "ModuleOp"> {
    let summary = "Lowers bitwise operations to signless integers";
    let description = [{
        This pass ensures that all bitwise operations are performed on signless
        integers, making all bit casts explicit.

        The affected operations are `bit.cmp`, `bit.and`, `bit.or`, `bit.xor`,
        `bit.shl`, `bit.shr`, `bit.count`, `bit.clz` and `bit.ctz`.
    }];

    let constructor = "mlir::bit::createLowerBitwisePass()";
}

#endif
