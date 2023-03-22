// RUN: base2-opt %s --canonicalize | FileCheck %s

//===----------------------------------------------------------------------===//
// value_cast
//===----------------------------------------------------------------------===//

func.func @value_cast_splat() -> tensor<3xsi8> {
    // CHECK-DAG: %[[RET:.+]] = bit.constant dense<127> : tensor<3xsi8>
    %cst = bit.constant dense<240> : tensor<3xui8>
    %0 = base2.cast %cst : tensor<3xui8> to nearest tensor<3xsi8>
    // return %[[RET]]
    return %0 : tensor<3xsi8>
}

func.func @value_cast_dense() -> tensor<3xsi8> {
    // CHECK-DAG: %[[RET:.+]] = bit.constant dense<[0, 100, 127]> : tensor<3xsi8>
    %cst = bit.constant dense<[0, 100, 240]> : tensor<3xui8>
    %0 = base2.cast %cst : tensor<3xui8> to nearest tensor<3xsi8>
    // return %[[RET]]
    return %0 : tensor<3xsi8>
}

//===----------------------------------------------------------------------===//
// cmp
//===----------------------------------------------------------------------===//

func.func @cmp_splat() -> tensor<3xi1> {
    // CHECK-DAG: %[[RET:.+]] = bit.constant dense<true> : tensor<3xi1>
    %lhs = bit.constant dense<127> : tensor<3xsi8>
    %rhs = bit.constant dense<120> : tensor<3xsi8>
    %0 = base2.cmp oge %lhs, %rhs : tensor<3xsi8>
    // return %[[RET]]
    return %0 : tensor<3xi1>
}

func.func @cmp_dense() -> tensor<3xi1> {
    // CHECK-DAG: %[[RET:.+]] = bit.constant dense<[false, true, true]> : tensor<3xi1>
    %lhs = bit.constant dense<[0, 120, 127]> : tensor<3xsi8>
    %rhs = bit.constant dense<120> : tensor<3xsi8>
    %0 = base2.cmp oge %lhs, %rhs : tensor<3xsi8>
    // return %[[RET]]
    return %0 : tensor<3xi1>
}

//===----------------------------------------------------------------------===//
// add
//===----------------------------------------------------------------------===//

func.func @add_splat() -> tensor<3xsi8> {
    // CHECK-DAG: %[[RET:.+]] = bit.constant dense<42> : tensor<3xsi8>
    %lhs = bit.constant dense<12> : tensor<3xsi8>
    %rhs = bit.constant dense<30> : tensor<3xsi8>
    %0 = base2.add %lhs, %rhs : tensor<3xsi8>
    // return %[[RET]]
    return %0 : tensor<3xsi8>
}

func.func @add_dense() -> tensor<3xsi8> {
    // CHECK-DAG: %[[RET:.+]] = bit.constant dense<[-128, 42, 127]> : tensor<3xsi8>
    %lhs = bit.constant dense<[-10, 12, 100]> : tensor<3xsi8>
    %rhs = bit.constant dense<[-128, 30, 28]> : tensor<3xsi8>
    %0 = base2.add %lhs, %rhs : nearest tensor<3xsi8>
    // return %[[RET]]
    return %0 : tensor<3xsi8>
}
