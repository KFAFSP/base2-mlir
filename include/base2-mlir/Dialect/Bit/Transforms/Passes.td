//===- Passes.td -------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the Bit passes.
//
//===----------------------------------------------------------------------===//

#ifndef BIT_PASSES
#define BIT_PASSES

include "mlir/Pass/PassBase.td"

def LowerFunnelShift : Pass<"lower-funnel-shift", "ModuleOp"> {
    let summary = "Lowers funnel shift operations to shift and or";
    let description = [{
        This pass implements a workaround for the missing LLVM dialect
        intrinsics `llvm.fsh?.*` funnel shift intrinsics. It replaces these
        shifts using normal shifts and bitwise or.
    }];

    let constructor = "mlir::bit::createLowerFunnelShiftPass()";
}

#endif
