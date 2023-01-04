// RUN: base2-opt --canonicalize %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Canonicalization (#base2.* -> builtin)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @builtin_bits_i16(
func.func @builtin_bits_i16() -> (i16, i16, i16, i16, i16) {
    // CHECK-DAG: %[[RET:.+]] = base2.constant 16 : i16
    %0 = base2.constant #base2.bits<"0b0000000000010000"> : i16
    %1 = base2.constant #base2.bits<"0b10000/16"> : i16
    %2 = base2.constant #base2.bits<"0b10000000000010000/16"> : i16
    %3 = base2.constant #base2.bits<"0x10/16"> : i16
    %4 = base2.constant #base2.bits<"0x10010/16"> : i16
    // CHECK: return %[[RET]], %[[RET]], %[[RET]], %[[RET]], %[[RET]]
    return %0, %1, %2, %3, %4 : i16, i16, i16, i16, i16
}

// CHECK-LABEL: func.func @builtin_bits_f32(
func.func @builtin_bits_f32() -> (f32, f32) {
    // CHECK-DAG: %[[RET:.+]] = base2.constant 1.342000e+01 : f32
    %0 = base2.constant #base2.bits<"0b_0_10000010_10101101011100001010010/32"> : f32
    %1 = base2.constant #base2.bits<"0x4156b852/32"> : f32
    // CHECK: return %[[RET]], %[[RET]]
    return %0, %1 : f32, f32
}

// CHECK-LABEL: func.func @builtin_dense_bits_i1(
func.func @builtin_dense_bits_i1() -> (tensor<3xi1>, tensor<3xi1>, tensor<10xi1>) {
    // CHECK-DAG: %[[RET0:.+]] = base2.constant dense<true> : tensor<3xi1>
    %0 = base2.constant #base2.dense_bits<tensor<3xi1> = dense<true>>
    // CHECK-DAG: %[[RET1:.+]] = base2.constant dense<[false, true, true]> : tensor<3xi1>
    %1 = base2.constant #base2.dense_bits<tensor<3xi1> = dense<[false, true, true]>>
    // CHECK-DAG: %[[RET2:.+]] = base2.constant dense<[false, true, false, true, false, true, false, true, false, true]> : tensor<10xi1>
    %2 = base2.constant #base2.dense_bits<tensor<10xi1> = dense<"0xAA02">>
    // CHECK: return %[[RET0]], %[[RET1]], %[[RET2]]
    return %0, %1, %2 : tensor<3xi1>, tensor<3xi1>, tensor<10xi1>
}

// CHECK-LABEL: func.func @builtin_dense_bits_i16(
func.func @builtin_dense_bits_i16() -> (tensor<3xi16>, tensor<3xi16>, tensor<3xi16>) {
    // CHECK-DAG: %[[RET0:.+]] = base2.constant dense<1> : tensor<3xi16>
    %0 = base2.constant #base2.dense_bits<tensor<3xi16> = dense<1>>
    // CHECK-DAG: %[[RET1:.+]] = base2.constant dense<[1, 2, 3]> : tensor<3xi16>
    %1 = base2.constant #base2.dense_bits<tensor<3xi16> = dense<[1, 2, 3]>>
    %2 = base2.constant #base2.dense_bits<tensor<3xi16> = dense<"0x010002000300">>
    // CHECK: return %[[RET0]], %[[RET1]], %[[RET1]]
    return %0, %1, %2 : tensor<3xi16>, tensor<3xi16>, tensor<3xi16>
}

// CHECK-LABEL: func.func @builtin_dense_bits_f16(
func.func @builtin_dense_bits_f16() -> (tensor<3xf16>, tensor<3xf16>, tensor<3xf16>) {
    // CHECK-DAG: %[[RET0:.+]] = base2.constant dense<1.342190e+01> : tensor<3xf16>
    %0 = base2.constant #base2.dense_bits<tensor<3xf16> = dense<"0x4AB6">>
    // CHECK-DAG: %[[RET1:.+]] = base2.constant dense<[1.342190e+01, 1.432030e+01, 4.321880e+01]> : tensor<3xf16>
    %1 = base2.constant #base2.dense_bits<tensor<3xf16> = dense<[0xB64A, 0x294B, 0x6751]>>
    %2 = base2.constant #base2.dense_bits<tensor<3xf16> = dense<"0x4AB64B295167">>
    // CHECK: return %[[RET0]], %[[RET1]], %[[RET1]]
    return %0, %1, %2 : tensor<3xf16>, tensor<3xf16>, tensor<3xf16>
}

//===----------------------------------------------------------------------===//
// Bitcasting
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @bitcast_bits_i16_si16(
func.func @bitcast_bits_i16_si16() -> si16 {
    // CHECK-DAG: %[[RET:.+]] = base2.constant 16 : si16
    %0 = base2.constant 16 : i16
    %1 = base2.bit_cast %0 : i16 to si16
    // CHECK: return %[[RET]]
    return %1 : si16
}

// CHECK-LABEL: func.func @bitcast_bits_f16_i16(
func.func @bitcast_bits_f16_i16() -> i16 {
    // CHECK-DAG: %[[RET:.+]] = base2.constant 19126 : i16
    %0 = base2.constant 1.342190e+01 : f16
    %1 = base2.bit_cast %0 : f16 to i16
    // CHECK: return %[[RET]]
    return %1 : i16
}

// CHECK-LABEL: func.func @bitcast_dense_bits_i16_si16(
func.func @bitcast_dense_bits_i16_si16() -> tensor<3xsi16> {
    // CHECK-DAG: %[[RET:.+]] = base2.constant dense<[1, 2, 3]> : tensor<3xsi16>
    %0 = base2.constant dense<[1, 2, 3]> : tensor<3xi16>
    %1 = base2.bit_cast %0 : tensor<3xi16> to tensor<3xsi16>
    // CHECK: return %[[RET]]
    return %1 : tensor<3xsi16>
}

// CHECK-LABEL: func.func @bitcast_splat_bits_i16_si16(
func.func @bitcast_splat_bits_i16_si16() -> tensor<3xsi16> {
    // CHECK-DAG: %[[RET:.+]] = base2.constant dense<1> : tensor<3xsi16>
    %0 = base2.constant dense<1> : tensor<3xi16>
    %1 = base2.bit_cast %0 : tensor<3xi16> to tensor<3xsi16>
    // CHECK: return %[[RET]]
    return %1 : tensor<3xsi16>
}

// CHECK-LABEL: func.func @bitcast_dense_bits_f16_i16(
func.func @bitcast_dense_bits_f16_i16() -> tensor<3xi16> {
    // CHECK-DAG: %[[RET:.+]] = base2.constant dense<[19126, 19241, 20839]> : tensor<3xi16>
    %0 = base2.constant dense<[1.342190e+01, 1.432030e+01, 4.321880e+01]> : tensor<3xf16>
    %1 = base2.bit_cast %0 : tensor<3xf16> to tensor<3xi16>
    // CHECK: return %[[RET]]
    return %1 : tensor<3xi16>
}

// CHECK-LABEL: func.func @bitcast_splat_bits_f16_i16(
func.func @bitcast_splat_bits_f16_i16() -> tensor<3xi16> {
    // CHECK-DAG: %[[RET:.+]] = base2.constant dense<19126> : tensor<3xi16>
    %0 = base2.constant dense<1.342190e+01> : tensor<3xf16>
    %1 = base2.bit_cast %0 : tensor<3xf16> to tensor<3xi16>
    // CHECK: return %[[RET]]
    return %1 : tensor<3xi16>
}
