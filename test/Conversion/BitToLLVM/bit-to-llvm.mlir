// RUN: base2-opt %s --convert-bit-to-llvm="index-bitwidth=64" --canonicalize | FileCheck %s

module {

//===----------------------------------------------------------------------===//
// constant
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @constant_scalar(
func.func @constant_scalar() -> (i32, si32, f32) {
    // CHECK-DAG: %[[F32:.+]] = llvm.mlir.constant(-6.2598534E+18 : f32) : f32
    // CHECK-DAG: %[[I32:.+]] = llvm.mlir.constant(-559038737 : i32) : i32
    %0 = bit.constant #bit.bits<"0xDEADBEEF"> : i32
    %1 = bit.constant #bit.bits<"0xDEADBEEF"> : si32
    // CHECK-DAG: %[[SI32:.+]] = builtin.unrealized_conversion_cast %[[I32]] : i32 to si32
    %2 = bit.constant #bit.bits<"0xDEADBEEF"> : f32
    // CHECK: return %[[I32]], %[[SI32]], %[[F32]]
    return %0, %1, %2 : i32, si32, f32
}

// CHECK-LABEL: func.func @constant_container(
func.func @constant_container() -> (vector<3xi32>, vector<3xsi32>, vector<3xf32>) {
    // CHECK-DAG: %[[F32:.+]] = llvm.mlir.constant(dense<-6.2598534E+18> : vector<3xf32>) : vector<3xf32>
    // CHECK-DAG: %[[I32:.+]] = llvm.mlir.constant(dense<-559038737> : vector<3xi32>) : vector<3xi32>
    %0 = bit.constant #bit.dense_bits<vector<3xi32> = dense<0xDEADBEEF>>
    %1 = bit.constant #bit.dense_bits<vector<3xsi32> = dense<0xDEADBEEF>>
    // CHECK-DAG: %[[SI32:.+]] = builtin.unrealized_conversion_cast %[[I32]] : vector<3xi32> to vector<3xsi32>
    %2 = bit.constant #bit.dense_bits<vector<3xf32> = dense<"0xDEADBEEF">>
    // CHECK: return %[[I32]], %[[SI32]], %[[F32]]
    return %0, %1, %2 : vector<3xi32>, vector<3xsi32>, vector<3xf32>
}

//===----------------------------------------------------------------------===//
// cast
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @cast(
// CHECK-SAME: %[[ARG0:.+]]: i32,
// CHECK-SAME: %[[ARG1:.+]]: si32,
// CHECK-SAME: %[[ARG2:.+]]: f32
func.func @cast(%arg0: i32, %arg1: si32, %arg2: f32) -> (si32, f32, f32, i32, si32) {
    %0 = bit.cast %arg0 : i32 to si32
    // CHECK-DAG: %[[RET0:.+]] = builtin.unrealized_conversion_cast %[[ARG0]] : i32 to si32
    %1 = bit.cast %arg0 : i32 to f32
    // CHECK-DAG: %[[RET1:.+]] = llvm.bitcast %[[ARG0]] : i32 to f32
    %2 = bit.cast %arg1 : si32 to f32
    // CHECK-DAG: %[[ARG1I:.+]] = builtin.unrealized_conversion_cast %[[ARG1]] : si32 to i32
    // CHECK-DAG: %[[RET2:.+]] = llvm.bitcast %[[ARG1I]] : i32 to f32
    %3 = bit.cast %arg2 : f32 to i32
    // CHECK-DAG: %[[RET3:.+]] = llvm.bitcast %[[ARG2]] : f32 to i32
    %4 = bit.cast %arg2 : f32 to si32
    // CHECK-DAG: %[[RET4I:.+]] = llvm.bitcast %[[ARG2]] : f32 to i32
    // CHECK-DAG: %[[RET4:.+]] = builtin.unrealized_conversion_cast %[[RET4I]] : i32 to si32
    // return %[[RET0]], %[[RET1]], %[[RET2]], %[[RET3]], %[[RET4]]
    return %0, %1, %2, %3, %4 : si32, f32, f32, i32, si32
}

//===----------------------------------------------------------------------===//
// cmp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @cmp(
// CHECK-SAME: %[[ARG0:.+]]: f32,
// CHECK-SAME: %[[ARG1:.+]]: f32
func.func @cmp(%arg0: f32, %arg1: f32) -> (i1, i1, i1, i1) {
    // CHECK-DAG: %[[FALSE:.+]] = llvm.mlir.constant(false) : i1
    // CHECK-DAG: %[[TRUE:.+]] = llvm.mlir.constant(true) : i1
    // CHECK-DAG: %[[ARG0I:.+]] = llvm.bitcast %[[ARG0]] : f32 to i32
    // CHECK-DAG: %[[ARG1I:.+]] = llvm.bitcast %[[ARG1]] : f32 to i32
    %0 = bit.cmp true %arg0, %arg1 : f32
    %1 = bit.cmp eq %arg0, %arg1 : f32
    // CHECK-DAG: %[[EQ:.+]] = llvm.icmp "eq" %[[ARG0I]], %[[ARG1I]]
    %2 = bit.cmp ne %arg0, %arg1 : f32
    // CHECK-DAG: %[[NE:.+]] = llvm.icmp "ne" %[[ARG0I]], %[[ARG1I]]
    %3 = bit.cmp false %arg0, %arg1 : f32
    // CHECK: return %[[TRUE]], %[[EQ]], %[[NE]], %[[FALSE]]
    return %0, %1, %2, %3 : i1, i1, i1, i1
}

//===----------------------------------------------------------------------===//
// select
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @select_scalar(
// CHECK-SAME: %[[ARG0:.+]]: i1,
// CHECK-SAME: %[[ARG1:.+]]: si32,
// CHECK-SAME: %[[ARG2:.+]]: si32
func.func @select_scalar(%arg0: i1, %arg1: si32, %arg2: si32) -> si32 {
    // CHECK-DAG: %[[ARG1I:.+]] = builtin.unrealized_conversion_cast %[[ARG1]] : si32 to i32
    // CHECK-DAG: %[[ARG2I:.+]] = builtin.unrealized_conversion_cast %[[ARG2]] : si32 to i32
    %0 = bit.select %arg0, %arg1, %arg2 : si32
    // CHECK-DAG: %[[SEL:.+]] = llvm.select %[[ARG0]], %[[ARG1I]], %[[ARG2I]]
    // CHECK-DAG: %[[RET:.+]] = builtin.unrealized_conversion_cast %[[SEL]] : i32 to si32
    // CHECK: return %[[RET]]
    return %0 : si32
}

// CHECK-LABEL: func.func @select_zip(
// CHECK-SAME: %[[ARG0:.+]]: vector<3xi1>,
// CHECK-SAME: %[[ARG1:.+]]: vector<3xsi32>,
// CHECK-SAME: %[[ARG2:.+]]: vector<3xsi32>
func.func @select_zip(%arg0: vector<3xi1>, %arg1: vector<3xsi32>, %arg2: vector<3xsi32>)
        -> vector<3xsi32>
{
    // CHECK-DAG: %[[ARG1I:.+]] = builtin.unrealized_conversion_cast %[[ARG1]] : vector<3xsi32> to vector<3xi32>
    // CHECK-DAG: %[[ARG2I:.+]] = builtin.unrealized_conversion_cast %[[ARG2]] : vector<3xsi32> to vector<3xi32>
    %0 = bit.select %arg0, %arg1, %arg2 : vector<3xi1>, vector<3xsi32>
    // CHECK-DAG: %[[SEL:.+]] = llvm.select %[[ARG0]], %[[ARG1I]], %[[ARG2I]]
    // CHECK-DAG: %[[RET:.+]] = builtin.unrealized_conversion_cast %[[SEL]] : vector<3xi32> to vector<3xsi32>
    // CHECK: return %[[RET]]
    return %0 : vector<3xsi32>
}

//===----------------------------------------------------------------------===//
// Logic operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @logic(
// CHECK-SAME: %[[ARG0:.+]]: f32,
// CHECK-SAME: %[[ARG1:.+]]: f32
func.func @logic(%arg0: f32, %arg1: f32) -> (f32, f32, f32) {
    // CHECK-DAG: %[[CAST0:.+]] = llvm.bitcast %[[ARG0]] : f32 to i32
    // CHECK-DAG: %[[CAST1:.+]] = llvm.bitcast %[[ARG1]] : f32 to i32
    %0 = bit.and %arg0, %arg1 : f32
    // CHECK-DAG: %[[AND:.+]] = llvm.and %[[CAST0]], %[[CAST1]]
    // CHECK-DAG: %[[RES0:.+]] = llvm.bitcast %[[AND]] : i32 to f32
    %1 = bit.or %arg0, %arg1 : f32
    // CHECK-DAG: %[[OR:.+]] = llvm.or %[[CAST0]], %[[CAST1]]
    // CHECK-DAG: %[[RES1:.+]] = llvm.bitcast %[[OR]] : i32 to f32
    %2 = bit.xor %arg0, %arg1 : f32
    // CHECK-DAG: %[[XOR:.+]] = llvm.xor %[[CAST0]], %[[CAST1]]
    // CHECK-DAG: %[[RES2:.+]] = llvm.bitcast %[[XOR]] : i32 to f32
    // CHECK: return %[[RES0]], %[[RES1]], %[[RES2]]
    return %0, %1, %2 : f32, f32, f32
}

//===----------------------------------------------------------------------===//
// Shifting operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @shl(
// CHECK-SAME: %[[ARG0:.+]]: i32,
// CHECK-SAME: %[[ARG1:.+]]: index,
// CHECK-SAME: %[[ARG2:.+]]: i32
func.func @shl(%arg0: i32, %arg1: index, %arg2: i32) -> i32 {
    // CHECK-DAG: %[[WIDTH:.+]] = llvm.mlir.constant(32 : i64) : i64
    // CHECK-DAG: %[[ARG1I:.+]] = builtin.unrealized_conversion_cast %[[ARG1]] : index to i64
    // CHECK-DAG: %[[INVAMOUNT:.+]] = llvm.sub %0, %1
    // CHECK-DAG: %[[SHIFT:.+]] = llvm.intr.umin(%[[ARG1I]], %[[WIDTH]])
    // CHECK-DAG: %[[SHIFTI:.+]] = llvm.trunc %[[SHIFT]] : i64 to i32
    // CHECK-DAG: %[[SHL:.+]] = llvm.shl %[[ARG0]], %[[SHIFTI]]
    // CHECK-DAG: %[[INVSHIFT:.+]] = llvm.intr.umin(%[[INVAMOUNT]], %[[WIDTH]])
    // CHECK-DAG: %[[INVSHIFTI:.+]] = llvm.trunc %[[INVSHIFT]] : i64 to i32
    // CHECK-DAG: %[[SHR:.+]] = llvm.lshr %[[ARG2]], %[[INVSHIFTI]]
    // CHECK-DAG: %[[RET:.+]] = llvm.or %[[SHL]], %[[SHR]]
    %0 = bit.shl %arg0:%arg2, %arg1 : i32
    // CHECK: return %[[RET]]
    return %0 : i32
}

// CHECK-LABEL: func.func @shr(
// CHECK-SAME: %[[ARG0:.+]]: i32,
// CHECK-SAME: %[[ARG1:.+]]: index,
// CHECK-SAME: %[[ARG2:.+]]: i32
func.func @shr(%arg0: i32, %arg1: index, %arg2: i32) -> i32 {
    // CHECK-DAG: %[[WIDTH:.+]] = llvm.mlir.constant(32 : i64) : i64
    // CHECK-DAG: %[[ARG1I:.+]] = builtin.unrealized_conversion_cast %[[ARG1]] : index to i64
    // CHECK-DAG: %[[INVAMOUNT:.+]] = llvm.sub %0, %1
    // CHECK-DAG: %[[SHIFT:.+]] = llvm.intr.umin(%[[ARG1I]], %[[WIDTH]])
    // CHECK-DAG: %[[SHIFTI:.+]] = llvm.trunc %[[SHIFT]] : i64 to i32
    // CHECK-DAG: %[[SHR:.+]] = llvm.lshr %[[ARG0]], %[[SHIFTI]]
    // CHECK-DAG: %[[INVSHIFT:.+]] = llvm.intr.umin(%[[INVAMOUNT]], %[[WIDTH]])
    // CHECK-DAG: %[[INVSHIFTI:.+]] = llvm.trunc %[[INVSHIFT]] : i64 to i32
    // CHECK-DAG: %[[SHL:.+]] = llvm.shl %[[ARG2]], %[[INVSHIFTI]]
    // CHECK-DAG: %[[RET:.+]] = llvm.or %[[SHR]], %[[SHL]]
    %0 = bit.shr %arg2:%arg0, %arg1 : i32
    // CHECK: return %[[RET]]
    return %0 : i32
}

//===----------------------------------------------------------------------===//
// Scanning operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @scanning(
// CHECK-SAME: %[[ARG0:.+]]: i32
func.func @scanning(%arg0: i32) -> (index, index, index) {
    // CHECK-DAG: %[[FALSE:.+]] = llvm.mlir.constant(false) : i1
    %0 = bit.count %arg0 : i32
    // CHECK-DAG: %[[COUNTI:.+]] = llvm.intr.ctpop(%[[ARG0]])
    // CHECK-DAG: %[[COUNTEXT:.+]] = llvm.zext %[[COUNTI]] : i32 to i64
    // CHECK-DAG: %[[COUNT:.+]] = builtin.unrealized_conversion_cast %[[COUNTEXT]] : i64 to index
    %1 = bit.clz %arg0 : i32
    // CHECK-DAG: %[[CLZI:.+]] = "llvm.intr.ctlz"(%[[ARG0]], %[[FALSE]])
    // CHECK-DAG: %[[CLZ:.+]] = builtin.unrealized_conversion_cast %[[CLZI]] : i64 to index
    %2 = bit.ctz %arg0 : i32
    // CHECK-DAG: %[[CTZI:.+]] = "llvm.intr.cttz"(%[[ARG0]], %[[FALSE]])
    // CHECK-DAG: %[[CTZ:.+]] = builtin.unrealized_conversion_cast %[[CTZI]] : i64 to index
    // CHECK: return %[[COUNT]], %[[CLZ]], %[[CTZ]]
    return %0, %1, %2 : index, index, index
}

}
