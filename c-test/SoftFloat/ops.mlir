!scalar = !softfloat.sfloat

func.func @add_caller(%arg0 : !scalar, %arg1: !scalar, %arg2 : i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> !scalar attributes {llvm.emit_c_interface} {
    %result = softfloat.add %arg0, %arg1 ( %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 ) : !scalar

    return %result : !scalar
}

func.func @sub_caller(%arg0 : !scalar, %arg1: !scalar, %arg2 : i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> !scalar attributes {llvm.emit_c_interface} {
    %result = softfloat.sub %arg0, %arg1 ( %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 ) : !scalar

    return %result : !scalar
}

func.func @mul_caller(%arg0 : !scalar, %arg1: !scalar, %arg2 : i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> !scalar attributes {llvm.emit_c_interface} {
    %result = softfloat.mul %arg0, %arg1 ( %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 ) : !scalar

    return %result : !scalar
}

func.func @divg_caller(%arg0 : !scalar, %arg1: !scalar, %arg2 : i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> !scalar attributes {llvm.emit_c_interface} {
    %result = softfloat.divg %arg0, %arg1 ( %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 ) : !scalar

    return %result : !scalar
}

func.func @divsrt_caller(%arg0 : !scalar, %arg1: !scalar, %arg2 : i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> !scalar attributes {llvm.emit_c_interface} {
    %result = softfloat.divsrt %arg0, %arg1 ( %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 ) : !scalar

    return %result : !scalar
}

func.func @eq_caller(%arg0 : !scalar, %arg1: !scalar, %arg2 : i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> i1 attributes {llvm.emit_c_interface} {
    %result = softfloat.eq %arg0, %arg1 ( %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 ) : i1

    return %result : i1
}

func.func @le_caller(%arg0 : !scalar, %arg1: !scalar, %arg2 : i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> i1 attributes {llvm.emit_c_interface} {
    %result = softfloat.le %arg0, %arg1 ( %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 ) : i1

    return %result : i1
}

func.func @lt_caller(%arg0 : !scalar, %arg1: !scalar, %arg2 : i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> i1 attributes {llvm.emit_c_interface} {
    %result = softfloat.lt %arg0, %arg1 ( %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 ) : i1

    return %result : i1
}

func.func @ge_caller(%arg0 : !scalar, %arg1: !scalar, %arg2 : i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> i1 attributes {llvm.emit_c_interface} {
    %result = softfloat.ge %arg0, %arg1 ( %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 ) : i1

    return %result : i1
}

func.func @gt_caller(%arg0 : !scalar, %arg1: !scalar, %arg2 : i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> i1 attributes {llvm.emit_c_interface} {
    %result = softfloat.gt %arg0, %arg1 ( %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 ) : i1

    return %result : i1
}

func.func @ltgt_caller(%arg0 : !scalar, %arg1: !scalar, %arg2 : i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> i1 attributes {llvm.emit_c_interface} {
    %result = softfloat.ltgt %arg0, %arg1 ( %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 ) : i1

    return %result : i1
}

func.func @nan_caller(%arg0 : !scalar, %arg1 : i8, %arg2: i8, %arg3: i32, %arg4: i1, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i8) -> i1 attributes {llvm.emit_c_interface} {
    %result = softfloat.nan %arg0 ( %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8 ) : i1

    return %result : i1
}

func.func @cast_caller(%arg0 : !scalar, %arg1 : i8, %arg2: i8, %arg3: i32, %arg4: i1, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i8, %arg9 : i8, %arg10: i8, %arg11: i32, %arg12: i1, %arg13: i1, %arg14: i1, %arg15: i1, %arg16: i8) -> !scalar attributes {llvm.emit_c_interface} {
    %result = softfloat.cast %arg0 ( %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16 ) : !scalar

    return %result : !scalar
}

func.func @castfloat_caller(%arg0 : f64, %arg1 : i8, %arg2: i8, %arg3: i32, %arg4: i1, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i8) -> !scalar attributes {llvm.emit_c_interface} {
    %result = softfloat.castfloat %arg0 ( %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8 ) : !scalar

    return %result : !scalar
}

func.func @casttofloat_caller(%arg0 : !scalar, %arg1 : i8, %arg2: i8, %arg3: i32, %arg4: i1, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i8) -> f64 attributes {llvm.emit_c_interface} {
    %result = softfloat.casttofloat %arg0 ( %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8 ) : f64

    return %result : f64
}
