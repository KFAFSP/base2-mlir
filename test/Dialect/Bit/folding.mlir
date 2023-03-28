// RUN: base2-opt --canonicalize %s | FileCheck %s

//===----------------------------------------------------------------------===//
// cast
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @cast_constant(
func.func @cast_constant() -> i32 {
    // CHECK: %[[CST0:.+]] = bit.constant 7 : i32
    // CHECK: return %[[CST0]] : i32
    %0 = bit.constant #bit.bits<"0x00000007"> : si32
    %1 = bit.cast %0 : si32 to i32
    return %1 : i32
}

// CHECK-LABEL: func.func @cast_poison(
func.func @cast_poison() -> i32 {
    // CHECK: %[[POISON:.+]] = ub.poison : i32
    %0 = ub.poison : si32
    %1 = bit.cast %0 : si32 to i32
    // CHECK: return %[[POISON]]
    return %1 : i32
}

//===----------------------------------------------------------------------===//
// cmp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @cmp_scalar(
func.func @cmp_scalar() -> (i1, i1, i1, i1) {
    // CHECK-DAG: %[[TRUE:.+]] = bit.constant true
    // CHECK-DAG: %[[FALSE:.+]] = bit.constant false
    %lhs = bit.constant 3 : i64
    %rhs = bit.constant 4 : i64
    %0 = bit.cmp true %lhs, %rhs : i64
    %1 = bit.cmp eq %lhs, %rhs : i64
    %2 = bit.cmp ne %lhs, %rhs : i64
    %3 = bit.cmp false %lhs, %rhs : i64
    // CHECK: return %[[TRUE]], %[[FALSE]], %[[TRUE]], %[[FALSE]]
    return %0, %1, %2, %3 : i1, i1, i1, i1
}

func.func @cmp_container() -> (tensor<3xi1>, tensor<3xi1>, tensor<3xi1>, tensor<3xi1>) {
    // CHECK-DAG: %[[TRUE:.+]] = bit.constant dense<true> : tensor<3xi1>
    // CHECK-DAG: %[[FALSE:.+]] = bit.constant dense<false> : tensor<3xi1>
    // CHECK-DAG: %[[CST0:.+]] = bit.constant dense<[false, true, false]> : tensor<3xi1>
    // CHECK-DAG: %[[CST1:.+]] = bit.constant dense<[true, false, true]> : tensor<3xi1>
    %lhs = bit.constant dense<[1, 2, 3]> : tensor<3xi64>
    %rhs = bit.constant dense<[3, 2, 1]> : tensor<3xi64>
    %0 = bit.cmp true %lhs, %rhs : tensor<3xi64>
    %1 = bit.cmp eq %lhs, %rhs : tensor<3xi64>
    %2 = bit.cmp ne %lhs, %rhs : tensor<3xi64>
    %3 = bit.cmp false %lhs, %rhs : tensor<3xi64>
    // CHECK: return %[[TRUE]], %[[CST0]], %[[CST1]], %[[FALSE]]
    return %0, %1, %2, %3 : tensor<3xi1>, tensor<3xi1>, tensor<3xi1>, tensor<3xi1>
}

//===----------------------------------------------------------------------===//
// select
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @select_const_cond(
// CHECK-SAME: %[[ARG0:.+]]: i64,
// CHECK-SAME: %[[ARG1:.+]]: i64
func.func @select_const_cond(%arg0: i64, %arg1: i64) -> (i64, i64) {
    %false = bit.constant false
    %true = bit.constant true
    %0 = bit.select %false, %arg0, %arg1 : i64
    %1 = bit.select %true, %arg0, %arg1 : i64
    // CHECK: return %[[ARG1]], %[[ARG0]]
    return %0, %1 : i64, i64
}

// CHECK-LABEL: func.func @select_zip(
func.func @select_zip()
        -> (tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) {
    // CHECK-DAG: %[[CST0:.+]] = bit.constant dense<0> : tensor<4xi64>
    // CHECK-DAG: %[[CST1:.+]] = bit.constant dense<[1, 0, 0, 1]> : tensor<4xi64>
    // CHECK-DAG: %[[CST2:.+]] = bit.constant dense<[0, 1, 1, 0]> : tensor<4xi64>
    %c1 = bit.constant dense<[1, 0, 0, 1]> : tensor<4xi1>
    %c2 = bit.constant dense<0> : tensor<4xi1>
    %c3 = bit.constant dense<1> : tensor<4xi1>
    %splat0 = bit.constant dense<0> : tensor<4xi64>
    %splat1 = bit.constant dense<1> : tensor<4xi64>
    %dense = bit.constant dense<[0, 1, 1, 0]> : tensor<4xi64>
    %0 = bit.select %c2, %splat1, %splat0 : tensor<4xi1>, tensor<4xi64>
    %1 = bit.select %c1, %splat1, %splat0 : tensor<4xi1>, tensor<4xi64>
    %2 = bit.select %c3, %dense, %splat0 : tensor<4xi1>, tensor<4xi64>
    %3 = bit.select %c2, %splat1, %dense : tensor<4xi1>, tensor<4xi64>
    // CHECK: return %[[CST0]], %[[CST1]], %[[CST2]], %[[CST2]]
    return %0, %1, %2, %3 : tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>
}

//===----------------------------------------------------------------------===//
// and
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @and_short_circuit(
// CHECK-SAME: %[[ARG0:.+]]: i64
func.func @and_short_circuit(%arg0: i64) -> (i64, i64) {
    // CHECK-DAG: %[[CST0:.+]] = bit.constant 0 : i64
    %cst0 = bit.constant 0 : i64
    %cst1 = bit.constant -1 : i64
    %0 = bit.and %arg0, %cst0 : i64
    %1 = bit.and %arg0, %cst1 : i64
    // CHECK: return %[[CST0]], %[[ARG0]]
    return %0, %1 : i64, i64
}

// CHECK-LABEL: func.func @and(
func.func @and() -> i64 {
    // CHECK-DAG: %[[CST0:.+]] = bit.constant 240 : i64
    %lhs = bit.constant 0x00FF : i64
    %rhs = bit.constant 0x0FF0 : i64
    %0 = bit.and %lhs, %rhs : i64
    // CHECK: return %[[CST0]]
    return %0 : i64
}

//===----------------------------------------------------------------------===//
// or
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @or_short_circuit(
// CHECK-SAME: %[[ARG0:.+]]: i64
func.func @or_short_circuit(%arg0: i64) -> (i64, i64) {
    %cst0 = bit.constant 0 : i64
    // CHECK-DAG: %[[CST0:.+]] = bit.constant -1 : i64
    %cst1 = bit.constant -1 : i64
    %0 = bit.or %arg0, %cst0 : i64
    %1 = bit.or %arg0, %cst1 : i64
    // CHECK: return %[[ARG0]], %[[CST0]]
    return %0, %1 : i64, i64
}

// CHECK-LABEL: func.func @or(
func.func @or() -> i64 {
    // CHECK-DAG: %[[CST0:.+]] = bit.constant 4095 : i64
    %lhs = bit.constant 0x00FF : i64
    %rhs = bit.constant 0x0FF0 : i64
    %0 = bit.or %lhs, %rhs : i64
    // CHECK: return %[[CST0]]
    return %0 : i64
}

//===----------------------------------------------------------------------===//
// xor
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @xor_short_circuit(
// CHECK-SAME: %[[ARG0:.+]]: i64
func.func @xor_short_circuit(%arg0: i64) -> i64 {
    %cst0 = bit.constant 0 : i64
    %0 = bit.xor %arg0, %cst0 : i64
    // CHECK: return %[[ARG0]]
    return %0 : i64
}

// CHECK-LABEL: func.func @xor(
func.func @xor() -> i64 {
    // CHECK-DAG: %[[CST0:.+]] = bit.constant 3855 : i64
    %lhs = bit.constant 0x00FF : i64
    %rhs = bit.constant 0x0FF0 : i64
    %0 = bit.xor %lhs, %rhs : i64
    // CHECK: return %[[CST0]]
    return %0 : i64
}

//===----------------------------------------------------------------------===//
// shl
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @shl_trivial(
// CHECK-SAME: %[[ARG0:.+]]: i64
func.func @shl_trivial(%arg0: i64) -> (i64, i64, i64) {
    // CHECK-DAG: %[[CST0:.+]] = bit.constant 0 : i64
    %cst0 = index.constant 0
    %cst1 = index.constant 64
    %cst2 = index.constant 128
    %0 = bit.shl %arg0, %cst0 : i64
    %1 = bit.shl %arg0, %cst1 : i64
    %2 = bit.shl %arg0:%arg0, %cst2 : i64
    // CHECK: return %[[ARG0]], %[[CST0]], %[[CST0]]
    return %0, %1, %2 : i64, i64, i64
}

// CHECK-LABEL: func.func @shl(
func.func @shl() -> (i16, i16) {
    // CHECK-DAG: %[[SHL:.+]] = bit.constant -17200 : i16
    // CHECK-DAG: %[[ROL:.+]] = bit.constant -17190 : i16
    %value = bit.constant 0xABCD : i16
    %cst0 = index.constant 4
    %0 = bit.shl %value, %cst0 : i16
    %1 = bit.shl %value:%value, %cst0 : i16
    // CHECK: return %[[SHL]], %[[ROL]]
    return %0, %1 : i16, i16
}

//===----------------------------------------------------------------------===//
// shr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @shr_trivial(
// CHECK-SAME: %[[ARG0:.+]]: i64
func.func @shr_trivial(%arg0: i64) -> (i64, i64, i64) {
    // CHECK-DAG: %[[CST0:.+]] = bit.constant 0 : i64
    %cst0 = index.constant 0
    %cst1 = index.constant 64
    %cst2 = index.constant 128
    %0 = bit.shr %arg0, %cst0 : i64
    %1 = bit.shr %arg0, %cst1 : i64
    %2 = bit.shr %arg0:%arg0, %cst2 : i64
    // CHECK: return %[[ARG0]], %[[CST0]], %[[CST0]]
    return %0, %1, %2 : i64, i64, i64
}

// CHECK-LABEL: func.func @shr(
func.func @shr() -> (i16, i16) {
    // CHECK-DAG: %[[SHR:.+]] = bit.constant 2748 : i16
    // CHECK-DAG: %[[ROR:.+]] = bit.constant -9540 : i16
    %value = bit.constant 0xABCD : i16
    %cst0 = index.constant 4
    %0 = bit.shr %value, %cst0 : i16
    %1 = bit.shr %value:%value, %cst0 : i16
    // CHECK: return %[[SHR]], %[[ROR]]
    return %0, %1 : i16, i16
}

//===----------------------------------------------------------------------===//
// count
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @count(
func.func @count() -> (index, index, index) {
    // CHECK-DAG: %[[CST0:.+]] = index.constant 0
    // CHECK-DAG: %[[CST1:.+]] = index.constant 16
    // CHECK-DAG: %[[CST2:.+]] = index.constant 32
    %cst0 = bit.constant 0 : i16
    %cst1 = bit.constant -1 : i16
    %cst2 = bit.constant -1 : i32
    %0 = bit.count %cst0 : i16
    %1 = bit.count %cst1 : i16
    %2 = bit.count %cst2 : i32
    // CHECK: return %[[CST0]], %[[CST1]], %[[CST2]]
    return %0, %1, %2 : index, index, index
}

//===----------------------------------------------------------------------===//
// clz
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @clz(
func.func @clz() -> (index, index, index) {
    // CHECK-DAG: %[[CST0:.+]] = index.constant 0
    // CHECK-DAG: %[[CST1:.+]] = index.constant 3
    // CHECK-DAG: %[[CST2:.+]] = index.constant 16
    %cst0 = bit.constant -1 : i16
    %cst1 = bit.constant 0x1000 : i16
    %cst2 = bit.constant 0 : i16
    %0 = bit.clz %cst0 : i16
    %1 = bit.clz %cst1 : i16
    %2 = bit.clz %cst2 : i16
    // CHECK: return %[[CST0]], %[[CST1]], %[[CST2]]
    return %0, %1, %2 : index, index, index
}

//===----------------------------------------------------------------------===//
// ctz
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @ctz(
func.func @ctz() -> (index, index, index) {
    // CHECK-DAG: %[[CST0:.+]] = index.constant 0
    // CHECK-DAG: %[[CST1:.+]] = index.constant 3
    // CHECK-DAG: %[[CST2:.+]] = index.constant 16
    %cst0 = bit.constant -1 : i16
    %cst1 = bit.constant 0x0008 : i16
    %cst2 = bit.constant 0 : i16
    %0 = bit.ctz %cst0 : i16
    %1 = bit.ctz %cst1 : i16
    %2 = bit.ctz %cst2 : i16
    // CHECK: return %[[CST0]], %[[CST1]], %[[CST2]]
    return %0, %1, %2 : index, index, index
}
