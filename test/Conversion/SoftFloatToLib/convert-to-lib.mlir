// RUN: base2-opt %s -convert-softfloat-to-lib | FileCheck %s

// CHECK-LABEL: func @add_caller
// CHECK-SAME: %[[ARG0:.*]]: i64, %[[ARG1:.*]]: i64
// CHECK-SAME: %[[ARG2:.*]]: i8, %[[ARG3:.*]]: i8, %[[ARG4:.*]]: i32
// CHECK-SAME: %[[ARG5:.*]]: i1, %[[ARG6:.*]]: i1, %[[ARG7:.*]]: i1, %[[ARG8:.*]]: i1, %[[ARG9:.*]]: i8
// CHECK-SAME: -> i64
func.func @add_caller(%arg0 : !softfloat.sfloat, %arg1: !softfloat.sfloat, %arg2 : i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> !softfloat.sfloat {
    %result = softfloat.add %arg0, %arg1 ( %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 ) : !softfloat.sfloat

    return %result : !softfloat.sfloat
}
// CHECK-DAG: %[[RESULT:.*]] = call @__float_add(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[ARG4]], %[[ARG5]], %[[ARG6]], %[[ARG7]], %[[ARG8]], %[[ARG9]]) : (i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i64
// CHECK: return %[[RESULT]] : i64

// CHECK-LABEL: func @sub_caller
// CHECK-SAME: %[[ARG0:.*]]: i64, %[[ARG1:.*]]: i64
// CHECK-SAME: %[[ARG2:.*]]: i8, %[[ARG3:.*]]: i8, %[[ARG4:.*]]: i32
// CHECK-SAME: %[[ARG5:.*]]: i1, %[[ARG6:.*]]: i1, %[[ARG7:.*]]: i1, %[[ARG8:.*]]: i1, %[[ARG9:.*]]: i8
// CHECK-SAME: -> i64
func.func @sub_caller(%arg0 : !softfloat.sfloat, %arg1: !softfloat.sfloat, %arg2 : i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> !softfloat.sfloat {
    %result = softfloat.sub %arg0, %arg1 ( %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 ) : !softfloat.sfloat

    return %result : !softfloat.sfloat
}
// CHECK-DAG: %[[RESULT:.*]] = call @__float_sub(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[ARG4]], %[[ARG5]], %[[ARG6]], %[[ARG7]], %[[ARG8]], %[[ARG9]]) : (i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i64
// CHECK: return %[[RESULT]] : i64

// CHECK-LABEL: func @mul_caller
// CHECK-SAME: %[[ARG0:.*]]: i64, %[[ARG1:.*]]: i64
// CHECK-SAME: %[[ARG2:.*]]: i8, %[[ARG3:.*]]: i8, %[[ARG4:.*]]: i32
// CHECK-SAME: %[[ARG5:.*]]: i1, %[[ARG6:.*]]: i1, %[[ARG7:.*]]: i1, %[[ARG8:.*]]: i1, %[[ARG9:.*]]: i8
// CHECK-SAME: -> i64
func.func @mul_caller(%arg0 : !softfloat.sfloat, %arg1: !softfloat.sfloat, %arg2 : i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> !softfloat.sfloat {
    %result = softfloat.mul %arg0, %arg1 ( %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 ) : !softfloat.sfloat

    return %result : !softfloat.sfloat
}
// CHECK-DAG: %[[RESULT:.*]] = call @__float_mul(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[ARG4]], %[[ARG5]], %[[ARG6]], %[[ARG7]], %[[ARG8]], %[[ARG9]]) : (i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i64
// CHECK: return %[[RESULT]] : i64

// CHECK-LABEL: func @divsrt_caller
// CHECK-SAME: %[[ARG0:.*]]: i64, %[[ARG1:.*]]: i64
// CHECK-SAME: %[[ARG2:.*]]: i8, %[[ARG3:.*]]: i8, %[[ARG4:.*]]: i32
// CHECK-SAME: %[[ARG5:.*]]: i1, %[[ARG6:.*]]: i1, %[[ARG7:.*]]: i1, %[[ARG8:.*]]: i1, %[[ARG9:.*]]: i8
// CHECK-SAME: -> i64
func.func @divsrt_caller(%arg0 : !softfloat.sfloat, %arg1: !softfloat.sfloat, %arg2 : i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> !softfloat.sfloat {
    %result = softfloat.divsrt %arg0, %arg1 ( %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 ) : !softfloat.sfloat

    return %result : !softfloat.sfloat
}
// CHECK-DAG: %[[RESULT:.*]] = call @__float_divSRT4(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[ARG4]], %[[ARG5]], %[[ARG6]], %[[ARG7]], %[[ARG8]], %[[ARG9]]) : (i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i64
// CHECK: return %[[RESULT]] : i64

// CHECK-LABEL: func @divg_caller
// CHECK-SAME: %[[ARG0:.*]]: i64, %[[ARG1:.*]]: i64
// CHECK-SAME: %[[ARG2:.*]]: i8, %[[ARG3:.*]]: i8, %[[ARG4:.*]]: i32
// CHECK-SAME: %[[ARG5:.*]]: i1, %[[ARG6:.*]]: i1, %[[ARG7:.*]]: i1, %[[ARG8:.*]]: i1, %[[ARG9:.*]]: i8
// CHECK-SAME: -> i64
func.func @divg_caller(%arg0 : !softfloat.sfloat, %arg1: !softfloat.sfloat, %arg2 : i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> !softfloat.sfloat {
    %result = softfloat.divg %arg0, %arg1 ( %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 ) : !softfloat.sfloat

    return %result : !softfloat.sfloat
}
// CHECK-DAG: %[[RESULT:.*]] = call @__float_divG(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[ARG4]], %[[ARG5]], %[[ARG6]], %[[ARG7]], %[[ARG8]], %[[ARG9]]) : (i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i64
// CHECK: return %[[RESULT]] : i64

// CHECK-LABEL: func @eq_caller
// CHECK-SAME: %[[ARG0:.*]]: i64, %[[ARG1:.*]]: i64
// CHECK-SAME: %[[ARG2:.*]]: i8, %[[ARG3:.*]]: i8, %[[ARG4:.*]]: i32
// CHECK-SAME: %[[ARG5:.*]]: i1, %[[ARG6:.*]]: i1, %[[ARG7:.*]]: i1, %[[ARG8:.*]]: i1, %[[ARG9:.*]]: i8
// CHECK-SAME: -> i1
func.func @eq_caller(%arg0 : !softfloat.sfloat, %arg1: !softfloat.sfloat, %arg2 : i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> i1 {
    %result = softfloat.eq %arg0, %arg1 ( %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 ) : i1

    return %result : i1
}
// CHECK-DAG: %[[RESULT:.*]] = call @__float_eq(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[ARG4]], %[[ARG5]], %[[ARG6]], %[[ARG7]], %[[ARG8]], %[[ARG9]]) : (i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i1
// CHECK: return %[[RESULT]] : i1

// CHECK-LABEL: func @le_caller
// CHECK-SAME: %[[ARG0:.*]]: i64, %[[ARG1:.*]]: i64
// CHECK-SAME: %[[ARG2:.*]]: i8, %[[ARG3:.*]]: i8, %[[ARG4:.*]]: i32
// CHECK-SAME: %[[ARG5:.*]]: i1, %[[ARG6:.*]]: i1, %[[ARG7:.*]]: i1, %[[ARG8:.*]]: i1, %[[ARG9:.*]]: i8
// CHECK-SAME: -> i1
func.func @le_caller(%arg0 : !softfloat.sfloat, %arg1: !softfloat.sfloat, %arg2 : i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> i1 {
    %result = softfloat.le %arg0, %arg1 ( %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 ) : i1

    return %result : i1
}
// CHECK-DAG: %[[RESULT:.*]] = call @__float_le(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[ARG4]], %[[ARG5]], %[[ARG6]], %[[ARG7]], %[[ARG8]], %[[ARG9]]) : (i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i1
// CHECK: return %[[RESULT]] : i1

// CHECK-LABEL: func @lt_caller
// CHECK-SAME: %[[ARG0:.*]]: i64, %[[ARG1:.*]]: i64
// CHECK-SAME: %[[ARG2:.*]]: i8, %[[ARG3:.*]]: i8, %[[ARG4:.*]]: i32
// CHECK-SAME: %[[ARG5:.*]]: i1, %[[ARG6:.*]]: i1, %[[ARG7:.*]]: i1, %[[ARG8:.*]]: i1, %[[ARG9:.*]]: i8
// CHECK-SAME: -> i1
func.func @lt_caller(%arg0 : !softfloat.sfloat, %arg1: !softfloat.sfloat, %arg2 : i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> i1 {
    %result = softfloat.lt %arg0, %arg1 ( %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 ) : i1

    return %result : i1
}
// CHECK-DAG: %[[RESULT:.*]] = call @__float_lt(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[ARG4]], %[[ARG5]], %[[ARG6]], %[[ARG7]], %[[ARG8]], %[[ARG9]]) : (i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i1
// CHECK: return %[[RESULT]] : i1

// CHECK-LABEL: func @ge_caller
// CHECK-SAME: %[[ARG0:.*]]: i64, %[[ARG1:.*]]: i64
// CHECK-SAME: %[[ARG2:.*]]: i8, %[[ARG3:.*]]: i8, %[[ARG4:.*]]: i32
// CHECK-SAME: %[[ARG5:.*]]: i1, %[[ARG6:.*]]: i1, %[[ARG7:.*]]: i1, %[[ARG8:.*]]: i1, %[[ARG9:.*]]: i8
// CHECK-SAME: -> i1
func.func @ge_caller(%arg0 : !softfloat.sfloat, %arg1: !softfloat.sfloat, %arg2 : i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> i1 {
    %result = softfloat.ge %arg0, %arg1 ( %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 ) : i1

    return %result : i1
}
// CHECK-DAG: %[[RESULT:.*]] = call @__float_ge(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[ARG4]], %[[ARG5]], %[[ARG6]], %[[ARG7]], %[[ARG8]], %[[ARG9]]) : (i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i1
// CHECK: return %[[RESULT]] : i1

// CHECK-LABEL: func @gt_caller
// CHECK-SAME: %[[ARG0:.*]]: i64, %[[ARG1:.*]]: i64
// CHECK-SAME: %[[ARG2:.*]]: i8, %[[ARG3:.*]]: i8, %[[ARG4:.*]]: i32
// CHECK-SAME: %[[ARG5:.*]]: i1, %[[ARG6:.*]]: i1, %[[ARG7:.*]]: i1, %[[ARG8:.*]]: i1, %[[ARG9:.*]]: i8
// CHECK-SAME: -> i1
func.func @gt_caller(%arg0 : !softfloat.sfloat, %arg1: !softfloat.sfloat, %arg2 : i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> i1 {
    %result = softfloat.gt %arg0, %arg1 ( %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 ) : i1

    return %result : i1
}
// CHECK-DAG: %[[RESULT:.*]] = call @__float_gt(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[ARG4]], %[[ARG5]], %[[ARG6]], %[[ARG7]], %[[ARG8]], %[[ARG9]]) : (i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i1
// CHECK: return %[[RESULT]] : i1

// CHECK-LABEL: func @ltgt_caller
// CHECK-SAME: %[[ARG0:.*]]: i64, %[[ARG1:.*]]: i64
// CHECK-SAME: %[[ARG2:.*]]: i8, %[[ARG3:.*]]: i8, %[[ARG4:.*]]: i32
// CHECK-SAME: %[[ARG5:.*]]: i1, %[[ARG6:.*]]: i1, %[[ARG7:.*]]: i1, %[[ARG8:.*]]: i1, %[[ARG9:.*]]: i8
// CHECK-SAME: -> i1
func.func @ltgt_caller(%arg0 : !softfloat.sfloat, %arg1: !softfloat.sfloat, %arg2 : i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> i1 {
    %result = softfloat.ltgt %arg0, %arg1 ( %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 ) : i1

    return %result : i1
}
// CHECK-DAG: %[[RESULT:.*]] = call @__float_ltgt_quiet(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[ARG4]], %[[ARG5]], %[[ARG6]], %[[ARG7]], %[[ARG8]], %[[ARG9]]) : (i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i1
// CHECK: return %[[RESULT]] : i1

// CHECK-LABEL: func @nan_caller
// CHECK-SAME: %[[ARG0:.*]]: i64, %[[ARG1:.*]]: i64
// CHECK-SAME: %[[ARG2:.*]]: i8, %[[ARG3:.*]]: i8, %[[ARG4:.*]]: i32
// CHECK-SAME: %[[ARG5:.*]]: i1, %[[ARG6:.*]]: i1, %[[ARG7:.*]]: i1, %[[ARG8:.*]]: i1, %[[ARG9:.*]]: i8
// CHECK-SAME: -> i1
func.func @nan_caller(%arg0 : !softfloat.sfloat, %arg1 : i8, %arg2: i8, %arg3: i32, %arg4: i1, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i8) -> i1 {
    %result = softfloat.nan %arg0 ( %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8 ) : i1

    return %result : i1
}
// CHECK-DAG: %[[RESULT:.*]] = call @__float64_is_signaling_nan(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[ARG4]], %[[ARG5]], %[[ARG6]], %[[ARG7]], %[[ARG8]], %[[ARG9]]) : (i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i1
// CHECK: return %[[RESULT]] : i1
