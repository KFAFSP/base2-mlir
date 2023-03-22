// RUN: base2-opt --canonicalize %s | FileCheck %s

//===----------------------------------------------------------------------===//
// cast
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @cast_noop(
// CHECK-SAME: %[[ARG0:.+]]: si64
func.func @cast_noop(%arg0: si64) -> si64 {
    %0 = base2.cast %arg0 : si64 to si64
    // CHECK: return %[[ARG0]] : si64
    return %0 : si64
}

//===----------------------------------------------------------------------===//
// cmp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @cmp_false(
func.func @cmp_false(%arg0: ui64, %arg1: ui64) -> (i1, i1, i1, i1, i1, i1) {
    // CHECK-DAG: %[[FALSE:.+]] = bit.constant false
    %cst0 = bit.constant 0 : ui64
    %0 = base2.cmp false %arg0, %arg1 : ui64
    %1 = base2.cmp uno %arg0, %arg1 : ui64
    %2 = base2.cmp olt %arg0, %cst0 : ui64
    %3 = base2.cmp ult %arg0, %cst0 : ui64
    %4 = base2.cmp ogt %cst0, %arg0 : ui64
    %5 = base2.cmp ugt %cst0, %arg0 : ui64
    // CHECK: return %[[FALSE]], %[[FALSE]], %[[FALSE]], %[[FALSE]], %[[FALSE]], %[[FALSE]]
    return %0, %1, %2, %3, %4, %5 : i1, i1, i1, i1, i1, i1
}

// CHECK-LABEL: func.func @cmp_true(
func.func @cmp_true(%arg0: ui64, %arg1: ui64) -> (i1, i1, i1, i1, i1, i1) {
    // CHECK-DAG: %[[TRUE:.+]] = bit.constant true
    %cst0 = bit.constant 0 : ui64
    %0 = base2.cmp true %arg0, %arg1 : ui64
    %1 = base2.cmp ord %arg0, %arg1 : ui64
    %2 = base2.cmp oge %arg0, %cst0 : ui64
    %3 = base2.cmp uge %arg0, %cst0 : ui64
    %4 = base2.cmp ole %cst0, %arg0 : ui64
    %5 = base2.cmp ule %cst0, %arg0 : ui64
    // CHECK: return %[[TRUE]], %[[TRUE]], %[[TRUE]], %[[TRUE]], %[[TRUE]], %[[TRUE]]
    return %0, %1, %2, %3, %4, %5 : i1, i1, i1, i1, i1, i1
}

// CHECK-LABEL: func.func @cmp_nan_ord(
func.func @cmp_nan_ord(%arg0: f32) -> (i1, i1, i1, i1, i1, i1, i1) {
    // CHECK-DAG: %[[FALSE:.+]] = bit.constant false
    %nan = bit.constant 0xFFFFFFFF : f32
    %0 = base2.cmp oeq %arg0, %nan : f32
    %1 = base2.cmp ogt %arg0, %nan : f32
    %2 = base2.cmp oge %arg0, %nan : f32
    %3 = base2.cmp olt %arg0, %nan : f32
    %4 = base2.cmp ole %arg0, %nan : f32
    %5 = base2.cmp one %arg0, %nan : f32
    %6 = base2.cmp ord %arg0, %nan : f32
    // CHECK: return %[[FALSE]], %[[FALSE]], %[[FALSE]], %[[FALSE]], %[[FALSE]], %[[FALSE]]
    return %0, %1, %2, %3, %4, %5, %6 : i1, i1, i1, i1, i1, i1, i1
}

// CHECK-LABEL: func.func @cmp_nan_uno(
func.func @cmp_nan_uno(%arg0: f32) -> (i1, i1, i1, i1, i1, i1, i1) {
    // CHECK-DAG: %[[TRUE:.+]] = bit.constant true
    %nan = bit.constant 0xFFFFFFFF : f32
    %0 = base2.cmp uno %arg0, %nan : f32
    %1 = base2.cmp ueq %arg0, %nan : f32
    %2 = base2.cmp ugt %arg0, %nan : f32
    %3 = base2.cmp uge %arg0, %nan : f32
    %4 = base2.cmp ult %arg0, %nan : f32
    %5 = base2.cmp ule %arg0, %nan : f32
    %6 = base2.cmp une %arg0, %nan : f32
    // CHECK: return %[[TRUE]], %[[TRUE]], %[[TRUE]], %[[TRUE]], %[[TRUE]], %[[TRUE]]
    return %0, %1, %2, %3, %4, %5, %6 : i1, i1, i1, i1, i1, i1, i1
}

//===----------------------------------------------------------------------===//
// min
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @min_trivial(
// CHECK-SAME: %[[ARG0:.+]]: i64
func.func @min_trivial(%arg0: i64) -> i64 {
    %0 = base2.min %arg0, %arg0 : i64
    // CHECK: return %[[ARG0]] : i64
    return %0 : i64
}

// CHECK-LABEL: func.func @min_ui_zero(
func.func @min_ui_zero(%arg0: ui64) -> (ui64, ui64) {
    // CHECK-DAG: %[[ZERO:.+]] = bit.constant 0 : ui64
    %cst0 = bit.constant 0 : ui64
    %0 = base2.min %arg0, %cst0 : ui64
    %1 = base2.min %cst0, %arg0 : ui64
    // RETURN %[[ZERO]], %[[ZERO]]
    return %0, %1 : ui64, ui64
}

// CHECK-LABEL: func.func @min_nan(
// CHECK-SAME: %[[ARG0:.+]]: f32
func.func @min_nan(%arg0: f32) -> (f32, f32) {
    %nan = bit.constant 0xFFFFFFFF : f32
    %0 = base2.min %arg0, %nan : f32
    %1 = base2.min %nan, %arg0 : f32
    // CHECK: return %[[ARG0]], %[[ARG0]]
    return %0, %1 : f32, f32
}

//===----------------------------------------------------------------------===//
// max
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @max_trivial(
// CHECK-SAME: %[[ARG0:.+]]: i64
func.func @max_trivial(%arg0: i64) -> i64 {
    %0 = base2.max %arg0, %arg0 : i64
    // CHECK: return %[[ARG0]] : i64
    return %0 : i64
}

// CHECK-LABEL: func.func @max_ui_zero(
// CHECK-SAME: %[[ARG0:.+]]: ui64
func.func @max_ui_zero(%arg0: ui64) -> (ui64, ui64) {
    %cst0 = bit.constant 0 : ui64
    %0 = base2.max %arg0, %cst0 : ui64
    %1 = base2.max %cst0, %arg0 : ui64
    // return %[[ARG0]], %[[ARG0]]
    return %0, %1 : ui64, ui64
}

// CHECK-LABEL: func.func @max_nan(
// CHECK-SAME: %[[ARG0:.+]]: f32
func.func @max_nan(%arg0: f32) -> (f32, f32) {
    %nan = bit.constant 0xFFFFFFFF : f32
    %0 = base2.max %arg0, %nan : f32
    %1 = base2.max %nan, %arg0 : f32
    // CHECK: return %[[ARG0]], %[[ARG0]]
    return %0, %1 : f32, f32
}

//===----------------------------------------------------------------------===//
// add
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @add_neutral(
// CHECK-SAME: %[[ARG0:.+]]: si64
func.func @add_neutral(%arg0: si64) -> (si64, si64) {
    %cst = bit.constant 0 : si64
    %0 = base2.add %arg0, %cst : si64
    %1 = base2.add %cst, %arg0 : si64
    // CHECK: return %[[ARG0]], %[[ARG0]]
    return %0, %1 : si64, si64
}

// CHECK-LABEL: func.func @add_nan(
// CHECK-SAME: %[[ARG0:.+]]: f32
func.func @add_nan(%arg0: f32) -> (f32, f32) {
    // CHECK-DAG: %[[NAN:.+]] = bit.constant 0xFFFFFFFF : f32
    %nan = bit.constant 0xFFFFFFFF : f32
    %0 = base2.add %arg0, %nan : f32
    %1 = base2.add %nan, %arg0 : f32
    // CHECK: return %[[NAN]], %[[NAN]]
    return %0, %1 : f32, f32
}

//===----------------------------------------------------------------------===//
// sub
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @sub_neutral(
// CHECK-SAME: %[[ARG0:.+]]: si64
func.func @sub_neutral(%arg0: si64) -> si64 {
    %cst = bit.constant 0 : si64
    %0 = base2.sub %arg0, %cst : si64
    // CHECK: return %[[ARG0]]
    return %0 : si64
}

// CHECK-LABEL: func.func @sub_nan(
// CHECK-SAME: %[[ARG0:.+]]: f32
func.func @sub_nan(%arg0: f32) -> (f32, f32) {
    // CHECK-DAG: %[[NAN:.+]] = bit.constant 0xFFFFFFFF : f32
    %nan = bit.constant 0xFFFFFFFF : f32
    %0 = base2.sub %arg0, %nan : f32
    %1 = base2.sub %nan, %arg0 : f32
    // CHECK: return %[[NAN]], %[[NAN]]
    return %0, %1 : f32, f32
}

//===----------------------------------------------------------------------===//
// mul
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @mul_neutral(
// CHECK-SAME: %[[ARG0:.+]]: si64
func.func @mul_neutral(%arg0: si64) -> (si64, si64) {
    %cst = bit.constant 1 : si64
    %0 = base2.mul %arg0, %cst : si64
    %1 = base2.mul %cst, %arg0 : si64
    // CHECK: return %[[ARG0]], %[[ARG0]]
    return %0, %1 : si64, si64
}

// CHECK-LABEL: func.func @mul_nan(
// CHECK-SAME: %[[ARG0:.+]]: f32
func.func @mul_nan(%arg0: f32) -> (f32, f32) {
    // CHECK-DAG: %[[NAN:.+]] = bit.constant 0xFFFFFFFF : f32
    %nan = bit.constant 0xFFFFFFFF : f32
    %0 = base2.mul %arg0, %nan : f32
    %1 = base2.mul %nan, %arg0 : f32
    // CHECK: return %[[NAN]], %[[NAN]]
    return %0, %1 : f32, f32
}

//===----------------------------------------------------------------------===//
// div
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @div_neutral(
// CHECK-SAME: %[[ARG0:.+]]: si64
func.func @div_neutral(%arg0: si64) -> si64 {
    %cst = bit.constant 1 : si64
    %0 = base2.div %arg0, %cst : si64
    // CHECK: return %[[ARG0]]
    return %0 : si64
}

// CHECK-LABEL: func.func @div_nan(
// CHECK-SAME: %[[ARG0:.+]]: f32
func.func @div_nan(%arg0: f32) -> (f32, f32) {
    // CHECK-DAG: %[[NAN:.+]] = bit.constant 0xFFFFFFFF : f32
    %nan = bit.constant 0xFFFFFFFF : f32
    %0 = base2.div %arg0, %nan : f32
    %1 = base2.div %nan, %arg0 : f32
    // CHECK: return %[[NAN]], %[[NAN]]
    return %0, %1 : f32, f32
}

//===----------------------------------------------------------------------===//
// mod
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @mod_nan(
// CHECK-SAME: %[[ARG0:.+]]: f32
func.func @mod_nan(%arg0: f32) -> (f32, f32) {
    // CHECK-DAG: %[[NAN:.+]] = bit.constant 0xFFFFFFFF : f32
    %nan = bit.constant 0xFFFFFFFF : f32
    %0 = base2.mod %arg0, %nan : f32
    %1 = base2.mod %nan, %arg0 : f32
    // CHECK: return %[[NAN]], %[[NAN]]
    return %0, %1 : f32, f32
}
