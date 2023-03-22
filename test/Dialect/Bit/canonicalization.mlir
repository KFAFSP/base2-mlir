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
