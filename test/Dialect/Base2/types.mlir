// RUN: base2-opt %s | FileCheck %s

//===----------------------------------------------------------------------===//
// fixed_point
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fixed_signless(
// CHECK-SAME: !base2.i11_3
func.func @fixed_signless(%arg0: !base2.fixed_point<11,3>)
        -> !base2.fixed_point<11,3> {
    return %arg0 : !base2.i11_3
}

// CHECK-LABEL: func.func @fixed_signed(
// CHECK-SAME: !base2.si12
func.func @fixed_signed(%arg0: !base2.fixed_point<signed 12>)
        -> !base2.fixed_point<signed 12> {
    return %arg0 : !base2.si12
}

// CHECK-LABEL: func.func @fixed_unsigned(
// CHECK-SAME: !base2.ui0_4
func.func @fixed_unsigned(%arg0: !base2.fixed_point<unsigned 0,4>)
        -> !base2.fixed_point<unsigned 0,4> {
    return %arg0 : !base2.ui0_4
}

//===----------------------------------------------------------------------===//
// ieee754
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @ieee754_simple(
// CHECK-SAME: !base2.f23_8
func.func @ieee754_simple(%arg0: !base2.ieee754<23, 8, 127>)
        -> !base2.ieee754<23, 8> {
    return %arg0 : !base2.f23_8_127
}

// CHECK-LABEL: func.func @ieee754_strange(
// CHECK-SAME: !base2.f11_4_8
func.func @ieee754_strange(%arg0: !base2.ieee754<11, 4, 8>)
        -> !base2.ieee754<11, 4, 8> {
    return %arg0 : !base2.f11_4_8
}
