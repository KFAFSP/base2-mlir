// RUN: base2-opt --lower-bitwise %s | FileCheck %s

module {

// CHECK-LABEL: func.func @cmp(
// CHECK-SAME: %[[ARG0:.+]]: si64,
// CHECK-SAME: %[[ARG1:.+]]: si64
func.func @cmp(%arg0: si64, %arg1: si64) -> (i1, i1) {
    // CHECK-DAG: %[[EQ0:.+]] = bit.cast %[[ARG0]] : si64 to i64
    // CHECK-DAG: %[[EQ1:.+]] = bit.cast %[[ARG1]] : si64 to i64
    // CHECK: %[[RES0:.+]] = bit.cmp eq %[[EQ0]], %[[EQ1]]
    %0 = bit.cmp eq %arg0, %arg1 : si64
    // CHECK-DAG: %[[NE0:.+]] = bit.cast %[[ARG0]] : si64 to i64
    // CHECK-DAG: %[[NE1:.+]] = bit.cast %[[ARG1]] : si64 to i64
    // CHECK: %[[RES1:.+]] = bit.cmp ne %[[NE0]], %[[NE1]]
    %1 = bit.cmp ne %arg0, %arg1 : si64
    // CHECK: return %[[RES0]], %[[RES1]]
    return %0, %1 : i1, i1
}

// CHECK-LABEL: func.func @logic(
// CHECK-SAME: %[[ARG0:.+]]: si64,
// CHECK-SAME: %[[ARG1:.+]]: si64
func.func @logic(%arg0: si64, %arg1: si64) -> (si64, si64, si64) {
    // CHECK-DAG: %[[AND0:.+]] = bit.cast %[[ARG0]] : si64 to i64
    // CHECK-DAG: %[[AND1:.+]] = bit.cast %[[ARG1]] : si64 to i64
    // CHECK: %[[AND:.+]] = bit.and %[[AND0]], %[[AND1]]
    %0 = bit.and %arg0, %arg1 : si64
    // CHECK: %[[RES0:.+]] = bit.cast %[[AND]] : i64 to si64
    // CHECK-DAG: %[[OR0:.+]] = bit.cast %[[ARG0]] : si64 to i64
    // CHECK-DAG: %[[OR1:.+]] = bit.cast %[[ARG1]] : si64 to i64
    // CHECK: %[[OR:.+]] = bit.or %[[OR0]], %[[OR1]]
    %1 = bit.or %arg0, %arg1 : si64
    // CHECK: %[[RES1:.+]] = bit.cast %[[OR]] : i64 to si64
    // CHECK-DAG: %[[XOR0:.+]] = bit.cast %[[ARG0]] : si64 to i64
    // CHECK-DAG: %[[XOR1:.+]] = bit.cast %[[ARG1]] : si64 to i64
    // CHECK: %[[XOR:.+]] = bit.xor %[[XOR0]], %[[XOR1]]
    %2 = bit.xor %arg0, %arg1 : si64
    // CHECK: %[[RES2:.+]] = bit.cast %[[XOR]] : i64 to si64
    // CHECK: return %[[RES0]], %[[RES1]], %[[RES2]]
    return %0, %1, %2 : si64, si64, si64
}

// CHECK-LABEL: func.func @shl(
// CHECK-SAME: %[[ARG0:.+]]: si64,
// CHECK-SAME: %[[ARG1:.+]]: index,
// CHECK-SAME: %[[ARG2:.+]]: si64
func.func @shl(%arg0: si64, %arg1: index, %arg2: si64) -> (si64, si64) {
    // CHECK-DAG: %[[SHL0:.+]] = bit.cast %[[ARG0]] : si64 to i64
    // CHECK: %[[SHL:.+]] = bit.shl %[[SHL0]], %[[ARG1]]
    %0 = bit.shl %arg0, %arg1 : si64
    // CHECK: %[[RES0:.+]] = bit.cast %[[SHL]] : i64 to si64
    // CHECK-DAG: %[[FSHL0:.+]] = bit.cast %[[ARG0]] : si64 to i64
    // CHECK-DAG: %[[FSHL1:.+]] = bit.cast %[[ARG2]] : si64 to i64
    // CHECK: %[[FSHL:.+]] = bit.shl %[[FSHL0]]:%[[FSHL1]], %[[ARG1]]
    %1 = bit.shl %arg0:%arg2, %arg1 : si64
    // CHECK: %[[RES1:.+]] = bit.cast %[[FSHL]] : i64 to si64
    // CHECK: return %[[RES0]], %[[RES1]]
    return %0, %1 : si64, si64
}

// CHECK-LABEL: func.func @shr(
// CHECK-SAME: %[[ARG0:.+]]: si64,
// CHECK-SAME: %[[ARG1:.+]]: index,
// CHECK-SAME: %[[ARG2:.+]]: si64
func.func @shr(%arg0: si64, %arg1: index, %arg2: si64) -> (si64, si64) {
    // CHECK-DAG: %[[SHR0:.+]] = bit.cast %[[ARG0]] : si64 to i64
    // CHECK: %[[SHR:.+]] = bit.shr %[[SHR0]], %[[ARG1]]
    %0 = bit.shr %arg0, %arg1 : si64
    // CHECK: %[[RES0:.+]] = bit.cast %[[SHR]] : i64 to si64
    // CHECK-DAG: %[[FSHR0:.+]] = bit.cast %[[ARG2]] : si64 to i64
    // CHECK-DAG: %[[FSHR1:.+]] = bit.cast %[[ARG0]] : si64 to i64
    // CHECK: %[[FSHR:.+]] = bit.shr %[[FSHR0]]:%[[FSHR1]], %[[ARG1]]
    %1 = bit.shr %arg2:%arg0, %arg1 : si64
    // CHECK: %[[RES1:.+]] = bit.cast %[[FSHR]] : i64 to si64
    // CHECK: return %[[RES0]], %[[RES1]]
    return %0, %1 : si64, si64
}

// CHECK-LABEL: func.func @scanning(
// CHECK-SAME: %[[ARG0:.+]]: si64
func.func @scanning(%arg0: si64) -> (index, index, index) {
    // CHECK-DAG: %[[COUNT0:.+]] = bit.cast %[[ARG0]] : si64 to i64
    // CHECK: %[[COUNT:.+]] = bit.count %[[COUNT0]]
    %0 = bit.count %arg0 : si64
    // CHECK-DAG: %[[CLZ0:.+]] = bit.cast %[[ARG0]] : si64 to i64
    // CHECK: %[[CLZ:.+]] = bit.clz %[[CLZ0]]
    %1 = bit.clz %arg0 : si64
    // CHECK-DAG: %[[CTZ0:.+]] = bit.cast %[[ARG0]] : si64 to i64
    // CHECK: %[[CTZ:.+]] = bit.ctz %[[CTZ0]]
    %2 = bit.ctz %arg0 : si64
    // CHECK: return %[[COUNT]], %[[CLZ]], %[[CTZ]]
    return %0, %1, %2 : index, index, index
}

}
