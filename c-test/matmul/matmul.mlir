#matmul_traits = {
    indexing_maps = [
        affine_map<(i,j,k) -> (i,k)>,
        affine_map<(i,j,k) -> (k,j)>,
        affine_map<(i,j,k) -> (i,j)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
}
!scalar = !softfloat.sfloat

func.func @cast_float(%a: f64, %exp_bits : i8, %frac_bits : i8, %exp_bias : i32) -> !scalar attributes {llvm.emit_c_interface} {
    %true = arith.constant true
    %sign = arith.constant -1 : i8

    %1 = softfloat.castfloat %a ( %exp_bits, %frac_bits, %exp_bias, %true, %true, %true, %true, %sign ) : !scalar

    return %1 : !scalar
}

func.func @cast_to_float(%a: !scalar, %exp_bits : i8, %frac_bits : i8, %exp_bias : i32) -> f64 attributes {llvm.emit_c_interface} {
    %true = arith.constant true
    %sign = arith.constant -1 : i8

    %1 = softfloat.casttofloat %a ( %exp_bits, %frac_bits, %exp_bias, %true, %true, %true, %true, %sign ) : f64

    return %1 : f64
}

func.func @test(%A: memref<11x11x11xf64>, %B: memref<11x11x11xf64>, %C: memref<11x11xf64>) attributes {llvm.emit_c_interface} {
    %A.mat = memref.collapse_shape %A [[0], [1, 2]] : memref<11x11x11xf64> into memref<11x121xf64>
    %B.mat = memref.collapse_shape %B [[0, 1], [2]] : memref<11x11x11xf64> into memref<121x11xf64>

    linalg.generic #matmul_traits
        ins(%A.mat, %B.mat: memref<11x121xf64>, memref<121x11xf64>)
        outs(%C: memref<11x11xf64>) {
    ^bb0(%a: f64, %b: f64, %c: f64):
        %0 = arith.mulf %a, %b : f64
        %1 = arith.addf %0, %c : f64
        linalg.yield %1 : f64
    }

    return
}

func.func @test_sf(%exp_bits : i8, %frac_bits : i8, %exp_bias : i32, %A: memref<11x11x11xi64>, %B: memref<11x11x11xi64>, %C: memref<11x11xi64>) attributes {llvm.emit_c_interface} {
    %A.mat = memref.collapse_shape %A [[0], [1, 2]] : memref<11x11x11xi64> into memref<11x121xi64>
    %B.mat = memref.collapse_shape %B [[0, 1], [2]] : memref<11x11x11xi64> into memref<121x11xi64>

    %true = arith.constant true
    %sign = arith.constant -1 : i8

    linalg.generic #matmul_traits
        ins(%A.mat, %B.mat: memref<11x121xi64>, memref<121x11xi64>)
        outs(%C: memref<11x11xi64>) {
    ^bb0(%a: i64, %b: i64, %c: i64):
        %a_cast = builtin.unrealized_conversion_cast %a : i64 to !scalar
        %b_cast = builtin.unrealized_conversion_cast %b : i64 to !scalar
        %c_cast = builtin.unrealized_conversion_cast %c : i64 to !scalar
        %0 = softfloat.mul %a_cast, %b_cast ( %exp_bits, %frac_bits, %exp_bias, %true, %true, %true, %true, %sign ) : !scalar
        %1 = softfloat.add %0, %c_cast ( %exp_bits, %frac_bits, %exp_bias, %true, %true, %true, %true, %sign ) : !scalar
        %2 = builtin.unrealized_conversion_cast %1 : !scalar to i64
        linalg.yield %2 : i64
    }

    return
}
