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

// CHECK-LABEL: func.func @cast_roundtrip(
// CHECK-SAME: %[[ARG0:.+]]: i64
func.func @cast_roundtrip(%arg0: i64) -> i64 {
    // CHECK: return %[[ARG0]] : i64
    %0 = bit.cast %arg0 : i64 to si64
    %1 = bit.cast %0 : si64 to ui64
    %2 = bit.cast %1 : ui64 to f64
    %3 = bit.cast %2 : f64 to i64
    return %3 : i64
}

//===----------------------------------------------------------------------===//
// cmp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @cmp_trivial(
func.func @cmp_trivial(%arg0: i64, %arg1: i64) -> (i1, i1) {
    // CHECK-DAG: %[[TRUE:.+]] = bit.constant true
    // CHECK-DAG: %[[FALSE:.+]] = bit.constant false
    %0 = bit.cmp true %arg0, %arg1 : i64
    %1 = bit.cmp false %arg0, %arg1 : i64
    // CHECK: return %[[TRUE]], %[[FALSE]]
    return %0, %1 : i1, i1
}

// CHECK-LABEL: func.func @cmp_trivial_scalar(
// CHECK-SAME: %[[ARG0:.+]]: i64
func.func @cmp_trivial_scalar(%arg0: i64) -> (i1, i1, i1, i1) {
    // CHECK-DAG: %[[TRUE:.+]] = bit.constant true
    // CHECK-DAG: %[[FALSE:.+]] = bit.constant false
    %0 = bit.cmp true %arg0, %arg0 : i64
    %1 = bit.cmp eq %arg0, %arg0 : i64
    %2 = bit.cmp ne %arg0, %arg0 : i64
    %3 = bit.cmp false %arg0, %arg0 : i64
    // CHECK: return %[[TRUE]], %[[TRUE]], %[[FALSE]], %[[FALSE]]
    return %0, %1, %2, %3 : i1, i1, i1, i1
}

// CHECK-LABEL: func.func @cmp_trivial_container(
// CHECK-SAME: %[[ARG0:.+]]: tensor<3xi64>
func.func @cmp_trivial_container(%arg0: tensor<3xi64>)
        -> (tensor<3xi1>, tensor<3xi1>, tensor<3xi1>, tensor<3xi1>) {
    // CHECK-DAG: %[[TRUE:.+]] = bit.constant dense<true> : tensor<3xi1>
    // CHECK-DAG: %[[FALSE:.+]] = bit.constant dense<false> : tensor<3xi1>
    %0 = bit.cmp true %arg0, %arg0 : tensor<3xi64>
    %1 = bit.cmp eq %arg0, %arg0 : tensor<3xi64>
    %2 = bit.cmp ne %arg0, %arg0 : tensor<3xi64>
    %3 = bit.cmp false %arg0, %arg0 : tensor<3xi64>
    // CHECK: return %[[TRUE]], %[[TRUE]], %[[FALSE]], %[[FALSE]]
    return %0, %1, %2, %3 : tensor<3xi1>, tensor<3xi1>, tensor<3xi1>, tensor<3xi1>
}

// CHECK-LABEL: func.func @cmp_bool_scalar(
// CHECK-SAME: %[[ARG0:.+]]: i1
func.func @cmp_bool_scalar(%arg0: i1) -> (i1, i1, i1, i1) {
    // CHECK-DAG: %[[TRUE:.+]] = bit.constant true
    %false = bit.constant false
    %true = bit.constant true
    %0 = bit.cmp eq %arg0, %false : i1
    // CHECK: %[[CMPL0:.+]] = bit.xor %arg0, %[[TRUE]] : i1
    %1 = bit.cmp ne %arg0, %false : i1
    %2 = bit.cmp eq %arg0, %true : i1
    %3 = bit.cmp ne %arg0, %true : i1
    // CHECK: %[[CMPL1:.+]] = bit.xor %arg0, %[[TRUE]] : i1
    // CHECK: return %[[CMPL0]], %[[ARG0]], %[[ARG0]], %[[CMPL1]]
    return %0, %1, %2, %3 : i1, i1, i1, i1
}

// CHECK-LABEL: func.func @cmp_bool_container(
// CHECK-SAME: %[[ARG0:.+]]: tensor<3xi1>
func.func @cmp_bool_container(%arg0: tensor<3xi1>)
    -> (tensor<3xi1>, tensor<3xi1>, tensor<3xi1>, tensor<3xi1>) {
    // CHECK-DAG: %[[TRUE:.+]] = bit.constant dense<true> : tensor<3xi1>
    %false = bit.constant dense<false> : tensor<3xi1>
    %true = bit.constant dense<true> : tensor<3xi1>
    %0 = bit.cmp eq %arg0, %false : tensor<3xi1>
    // CHECK: %[[CMPL0:.+]] = bit.xor %arg0, %[[TRUE]]
    %1 = bit.cmp ne %arg0, %false : tensor<3xi1>
    %2 = bit.cmp eq %arg0, %true : tensor<3xi1>
    %3 = bit.cmp ne %arg0, %true : tensor<3xi1>
    // CHECK: %[[CMPL1:.+]] = bit.xor %arg0, %[[TRUE]]
    // CHECK: return %[[CMPL0]], %[[ARG0]], %[[ARG0]], %[[CMPL1]]
    return %0, %1, %2, %3 : tensor<3xi1>, tensor<3xi1>, tensor<3xi1>, tensor<3xi1>
}

//===----------------------------------------------------------------------===//
// select
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @select_trivial(
// CHECK-SAME: %[[ARG0:.+]]: i1,
// CHECK-SAME: %[[ARG1:.+]]: i64
func.func @select_trivial(%arg0: i1, %arg1: i64) -> i64 {
    %0 = bit.select %arg0, %arg1, %arg1 : i64
    // CHECK: return %[[ARG1]]
    return %0 : i64
}

//===----------------------------------------------------------------------===//
// and
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @and_trivial(
// CHECK-SAME: %[[ARG0:.+]]: i64
func.func @and_trivial(%arg0: i64) -> i64 {
    %0 = bit.and %arg0, %arg0 : i64
    // CHECK: return %[[ARG0]]
    return %0 : i64
}

//===----------------------------------------------------------------------===//
// or
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @or_trivial(
// CHECK-SAME: %[[ARG0:.+]]: i64
func.func @or_trivial(%arg0: i64) -> i64 {
    %0 = bit.or %arg0, %arg0 : i64
    // CHECK: return %[[ARG0]]
    return %0 : i64
}

//===----------------------------------------------------------------------===//
// xor
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @xor_trivial_scalar(
// CHECK-SAME: %[[ARG0:.+]]: i64
func.func @xor_trivial_scalar(%arg0: i64) -> i64 {
    // CHECK-DAG: %[[CST0:.+]] = bit.constant 0 : i64
    %0 = bit.xor %arg0, %arg0 : i64
    // CHECK: return %[[CST0]]
    return %0 : i64
}

// CHECK-LABEL: func.func @xor_trivial_container(
func.func @xor_trivial_container(%arg0: tensor<3xi64>) -> tensor<3xi64> {
    // CHECK-DAG: %[[CST0:.+]] = bit.constant dense<0> : tensor<3xi64>
    %0 = bit.xor %arg0, %arg0 : tensor<3xi64>
    // CHECK: return %[[CST0]]
    return %0 : tensor<3xi64>
}
