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

// CHECK-LABEL: func.func @cast_poison_partial(
func.func @cast_poison_partial() -> tensor<3xi32> {
    // CHECK: %[[POISON:.+]] = ub.poison #ub.poison<"0000000000000005", bit(dense<[0, 1, 0]>)> : tensor<3xi32>
    %0 = ub.poison #ub.poison<"05", bit(dense<[0,1,2]>)> : tensor<3xsi32>
    %1 = bit.cast %0 : tensor<3xsi32> to tensor<3xi32>
    // CHECK: return %[[POISON]]
    return %1 : tensor<3xi32>
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

// CHECK-LABEL: func.func @cmp_poison_container(
func.func @cmp_poison_container()
        -> (tensor<3xi1>, tensor<3xi1>, tensor<3xi1>, tensor<3xi1>) {
    // CHECK-DAG: %[[TRUE:.+]] = bit.constant dense<true> : tensor<3xi1>
    // CHECK-DAG: %[[FALSE:.+]] = bit.constant dense<false> : tensor<3xi1>
    // CHECK-DAG: %[[POISON:.+]] = ub.poison #ub.poison<"0000000000000005", bit(dense<[false, true, false]>)> : tensor<3xi1>
    %lhs = ub.poison #ub.poison<"04", bit(dense<[0,1,2]>)> : tensor<3xi64>
    %rhs = ub.poison #ub.poison<"01", bit(dense<[0,1,2]>)> : tensor<3xi64>
    %0 = bit.cmp true %lhs, %rhs : tensor<3xi64>
    %1 = bit.cmp eq %lhs, %rhs : tensor<3xi64>
    %2 = bit.cmp eq %rhs, %lhs : tensor<3xi64>
    %3 = bit.cmp false %rhs, %lhs : tensor<3xi64>
    // CHECK: return %[[TRUE]], %[[POISON]], %[[POISON]], %[[FALSE]]
    return %0, %1, %2, %3 : tensor<3xi1>, tensor<3xi1>, tensor<3xi1>, tensor<3xi1>
}

//===----------------------------------------------------------------------===//
// select
//===----------------------------------------------------------------------===//

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

// CHECK-LABEL: func.func @select_poison(
func.func @select_poison() -> (tensor<3xi64>) {
    // CHECK-DAG: %[[POISON:.+]] = ub.poison #ub.poison<"0000000000000005", bit(dense<[0, 1, 0]>)> : tensor<3xi64>
    %cond = ub.poison #ub.poison<"05", bit(dense<[false, false, false]>)> : tensor<3xi1>
    %false = bit.constant dense<[1, 1, 1]> : tensor<3xi64>
    %true = bit.constant dense<[2, 2, 2]> : tensor<3xi64>
    %0 = bit.select %cond, %true, %false : tensor<3xi1>, tensor<3xi64>
    // CHECK: return %[[POISON]]
    return %0 : tensor<3xi64>
}

//===----------------------------------------------------------------------===//
// and
//===----------------------------------------------------------------------===//

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
