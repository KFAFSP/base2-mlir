// RUN: base2-opt --lower-bitwise-logic --cse %s | FileCheck %s

module {

// CHECK-LABEL: func.func @binary(
// CHECK-SAME: %[[ARG0:.+]]: si64,
// CHECK-SAME: %[[ARG1:.+]]: si64
func.func @binary(%arg0: si64, %arg1: si64) -> (si64, si64, si64) {
    // CHECK-DAG: %[[CAST0:.+]] = bit.cast %[[ARG0]] : si64 to i64
    // CHECK-DAG: %[[CAST1:.+]] = bit.cast %[[ARG1]] : si64 to i64
    // CHECK: %[[AND:.+]] = bit.and %[[CAST0]], %[[CAST1]]
    %0 = bit.and %arg0, %arg1 : si64
    // CHECK: %[[RES0:.+]] = bit.cast %[[AND]] : i64 to si64
    // CHECK: %[[OR:.+]] = bit.or %[[CAST0]], %[[CAST1]]
    %1 = bit.or %arg0, %arg1 : si64
    // CHECK: %[[RES1:.+]] = bit.cast %[[OR]] : i64 to si64
    // CHECK: %[[XOR:.+]] = bit.xor %[[CAST0]], %[[CAST1]]
    %2 = bit.xor %arg0, %arg1 : si64
    // CHECK: %[[RES2:.+]] = bit.cast %[[XOR]] : i64 to si64
    // CHECK: return %[[RES0]], %[[RES1]], %[[RES2]]
    return %0, %1, %2 : si64, si64, si64
}

}
