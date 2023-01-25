// RUN: base2-opt %s | base2-opt | FileCheck %s
// RUN: base2-opt %s --mlir-print-op-generic | base2-opt | FileCheck %s

// // CHECK-LABEL: test_add
func.func @test_add(%arg0 : !softfloat.sfloat, %arg1: !softfloat.sfloat, %arg2 : i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> !softfloat.sfloat {
    %0 = softfloat.add %arg0, %arg1 ( %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 ): !softfloat.sfloat
    return %0 : !softfloat.sfloat
}

// // CHECK-LABEL: test_sub
func.func @test_sub(%arg0 : !softfloat.sfloat, %arg1: !softfloat.sfloat, %arg2 : i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> !softfloat.sfloat {
    %0 = softfloat.sub %arg0, %arg1 ( %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 ): !softfloat.sfloat
    return %0 : !softfloat.sfloat
}

// // CHECK-LABEL: test_mul
func.func @test_mul(%arg0 : !softfloat.sfloat, %arg1: !softfloat.sfloat, %arg2 : i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> !softfloat.sfloat {
    %0 = softfloat.mul %arg0, %arg1 ( %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 ): !softfloat.sfloat
    return %0 : !softfloat.sfloat
}

// // CHECK-LABEL: test_divsrt
func.func @test_divsrt(%arg0 : !softfloat.sfloat, %arg1: !softfloat.sfloat, %arg2 : i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> !softfloat.sfloat {
    %0 = softfloat.divsrt %arg0, %arg1 ( %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 ): !softfloat.sfloat
    return %0 : !softfloat.sfloat
}

// // CHECK-LABEL: test_divgd
func.func @test_divgd(%arg0 : !softfloat.sfloat, %arg1: !softfloat.sfloat, %arg2 : i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> !softfloat.sfloat {
    %0 = softfloat.divg %arg0, %arg1 ( %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 ): !softfloat.sfloat
    return %0 : !softfloat.sfloat
}

// // CHECK-LABEL: test_eq
func.func @test_eq(%arg0 : !softfloat.sfloat, %arg1: !softfloat.sfloat, %arg2 : i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> i1 {
    %0 = softfloat.eq %arg0, %arg1 ( %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 ): i1
    return %0 : i1
}

// // CHECK-LABEL: test_le
func.func @test_le(%arg0 : !softfloat.sfloat, %arg1: !softfloat.sfloat, %arg2 : i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> i1 {
    %0 = softfloat.le %arg0, %arg1 ( %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 ): i1
    return %0 : i1
}

// // CHECK-LABEL: test_lt
func.func @test_lt(%arg0 : !softfloat.sfloat, %arg1: !softfloat.sfloat, %arg2 : i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> i1 {
    %0 = softfloat.lt %arg0, %arg1 ( %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 ): i1
    return %0 : i1
}

// // CHECK-LABEL: test_ge
func.func @test_ge(%arg0 : !softfloat.sfloat, %arg1: !softfloat.sfloat, %arg2 : i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> i1 {
    %0 = softfloat.ge %arg0, %arg1 ( %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 ): i1
    return %0 : i1
}

// // CHECK-LABEL: test_gt
func.func @test_gt(%arg0 : !softfloat.sfloat, %arg1: !softfloat.sfloat, %arg2 : i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> i1 {
    %0 = softfloat.gt %arg0, %arg1 ( %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 ): i1
    return %0 : i1
}

// // CHECK-LABEL: test_ltgt
func.func @test_ltgt(%arg0 : !softfloat.sfloat, %arg1: !softfloat.sfloat, %arg2 : i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> i1 {
    %0 = softfloat.ltgt %arg0, %arg1 ( %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 ): i1
    return %0 : i1
}

// // CHECK-LABEL: test_nan
func.func @test_nan(%arg0 : !softfloat.sfloat, %arg1: !softfloat.sfloat, %arg2 : i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> i1 {
    %0 = softfloat.nan %arg0 ( %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 ): i1
    return %0 : i1
}

// CHECK-LABEL: test_cast
func.func @test_cast(%arg0 : !softfloat.sfloat, %arg1 : i8, %arg2: i8, %arg3: i32, %arg4: i1, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i8, %arg9 : i8, %arg10: i8, %arg11: i32, %arg12: i1, %arg13: i1, %arg14: i1, %arg15: i1, %arg16: i8) -> !softfloat.sfloat {
    %0 = softfloat.cast %arg0 ( %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16 ) : !softfloat.sfloat
    return %0 : !softfloat.sfloat
}
