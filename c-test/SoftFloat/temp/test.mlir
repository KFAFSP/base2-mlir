module attributes {llvm.data_layout = ""} {
  llvm.func @__float_cast(i64, i8, i8, i32, i1, i1, i1, i1, i8, i8, i8, i32, i1, i1, i1, i1, i8) -> i64 attributes {sym_visibility = "private"}
  llvm.func @__float64_is_signaling_nan(i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i1 attributes {sym_visibility = "private"}
  llvm.func @__float_ltgt_quiet(i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i1 attributes {sym_visibility = "private"}
  llvm.func @__float_gt(i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i1 attributes {sym_visibility = "private"}
  llvm.func @__float_ge(i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i1 attributes {sym_visibility = "private"}
  llvm.func @__float_lt(i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i1 attributes {sym_visibility = "private"}
  llvm.func @__float_le(i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i1 attributes {sym_visibility = "private"}
  llvm.func @__float_eq(i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i1 attributes {sym_visibility = "private"}
  llvm.func @__float_divSRT4(i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i64 attributes {sym_visibility = "private"}
  llvm.func @__float_divG(i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i64 attributes {sym_visibility = "private"}
  llvm.func @__float_mul(i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i64 attributes {sym_visibility = "private"}
  llvm.func @__float_sub(i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i64 attributes {sym_visibility = "private"}
  llvm.func @__float_add(i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i64 attributes {sym_visibility = "private"}
  llvm.func @add_caller(%arg0: i64, %arg1: i64, %arg2: i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> i64 attributes {llvm.emit_c_interface} {
    %0 = llvm.call @__float_add(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : (i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i64
    llvm.return %0 : i64
  }
  llvm.func @_mlir_ciface_add_caller(%arg0: i64, %arg1: i64, %arg2: i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> i64 attributes {llvm.emit_c_interface} {
    %0 = llvm.call @add_caller(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : (i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i64
    llvm.return %0 : i64
  }
  llvm.func @sub_caller(%arg0: i64, %arg1: i64, %arg2: i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> i64 attributes {llvm.emit_c_interface} {
    %0 = llvm.call @__float_sub(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : (i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i64
    llvm.return %0 : i64
  }
  llvm.func @_mlir_ciface_sub_caller(%arg0: i64, %arg1: i64, %arg2: i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> i64 attributes {llvm.emit_c_interface} {
    %0 = llvm.call @sub_caller(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : (i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i64
    llvm.return %0 : i64
  }
  llvm.func @mul_caller(%arg0: i64, %arg1: i64, %arg2: i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> i64 attributes {llvm.emit_c_interface} {
    %0 = llvm.call @__float_mul(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : (i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i64
    llvm.return %0 : i64
  }
  llvm.func @_mlir_ciface_mul_caller(%arg0: i64, %arg1: i64, %arg2: i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> i64 attributes {llvm.emit_c_interface} {
    %0 = llvm.call @mul_caller(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : (i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i64
    llvm.return %0 : i64
  }
  llvm.func @divg_caller(%arg0: i64, %arg1: i64, %arg2: i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> i64 attributes {llvm.emit_c_interface} {
    %0 = llvm.call @__float_divG(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : (i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i64
    llvm.return %0 : i64
  }
  llvm.func @_mlir_ciface_divg_caller(%arg0: i64, %arg1: i64, %arg2: i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> i64 attributes {llvm.emit_c_interface} {
    %0 = llvm.call @divg_caller(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : (i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i64
    llvm.return %0 : i64
  }
  llvm.func @divsrt_caller(%arg0: i64, %arg1: i64, %arg2: i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> i64 attributes {llvm.emit_c_interface} {
    %0 = llvm.call @__float_divSRT4(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : (i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i64
    llvm.return %0 : i64
  }
  llvm.func @_mlir_ciface_divsrt_caller(%arg0: i64, %arg1: i64, %arg2: i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> i64 attributes {llvm.emit_c_interface} {
    %0 = llvm.call @divsrt_caller(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : (i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i64
    llvm.return %0 : i64
  }
  llvm.func @eq_caller(%arg0: i64, %arg1: i64, %arg2: i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> i1 attributes {llvm.emit_c_interface} {
    %0 = llvm.call @__float_eq(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : (i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i1
    llvm.return %0 : i1
  }
  llvm.func @_mlir_ciface_eq_caller(%arg0: i64, %arg1: i64, %arg2: i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> i1 attributes {llvm.emit_c_interface} {
    %0 = llvm.call @eq_caller(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : (i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i1
    llvm.return %0 : i1
  }
  llvm.func @le_caller(%arg0: i64, %arg1: i64, %arg2: i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> i1 attributes {llvm.emit_c_interface} {
    %0 = llvm.call @__float_le(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : (i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i1
    llvm.return %0 : i1
  }
  llvm.func @_mlir_ciface_le_caller(%arg0: i64, %arg1: i64, %arg2: i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> i1 attributes {llvm.emit_c_interface} {
    %0 = llvm.call @le_caller(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : (i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i1
    llvm.return %0 : i1
  }
  llvm.func @lt_caller(%arg0: i64, %arg1: i64, %arg2: i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> i1 attributes {llvm.emit_c_interface} {
    %0 = llvm.call @__float_lt(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : (i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i1
    llvm.return %0 : i1
  }
  llvm.func @_mlir_ciface_lt_caller(%arg0: i64, %arg1: i64, %arg2: i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> i1 attributes {llvm.emit_c_interface} {
    %0 = llvm.call @lt_caller(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : (i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i1
    llvm.return %0 : i1
  }
  llvm.func @ge_caller(%arg0: i64, %arg1: i64, %arg2: i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> i1 attributes {llvm.emit_c_interface} {
    %0 = llvm.call @__float_ge(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : (i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i1
    llvm.return %0 : i1
  }
  llvm.func @_mlir_ciface_ge_caller(%arg0: i64, %arg1: i64, %arg2: i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> i1 attributes {llvm.emit_c_interface} {
    %0 = llvm.call @ge_caller(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : (i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i1
    llvm.return %0 : i1
  }
  llvm.func @gt_caller(%arg0: i64, %arg1: i64, %arg2: i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> i1 attributes {llvm.emit_c_interface} {
    %0 = llvm.call @__float_gt(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : (i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i1
    llvm.return %0 : i1
  }
  llvm.func @_mlir_ciface_gt_caller(%arg0: i64, %arg1: i64, %arg2: i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> i1 attributes {llvm.emit_c_interface} {
    %0 = llvm.call @gt_caller(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : (i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i1
    llvm.return %0 : i1
  }
  llvm.func @ltgt_caller(%arg0: i64, %arg1: i64, %arg2: i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> i1 attributes {llvm.emit_c_interface} {
    %0 = llvm.call @__float_ltgt_quiet(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : (i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i1
    llvm.return %0 : i1
  }
  llvm.func @_mlir_ciface_ltgt_caller(%arg0: i64, %arg1: i64, %arg2: i8, %arg3: i8, %arg4: i32, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i8) -> i1 attributes {llvm.emit_c_interface} {
    %0 = llvm.call @ltgt_caller(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : (i64, i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i1
    llvm.return %0 : i1
  }
  llvm.func @nan_caller(%arg0: i64, %arg1: i8, %arg2: i8, %arg3: i32, %arg4: i1, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i8) -> i1 attributes {llvm.emit_c_interface} {
    %0 = llvm.call @__float64_is_signaling_nan(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8) : (i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i1
    llvm.return %0 : i1
  }
  llvm.func @_mlir_ciface_nan_caller(%arg0: i64, %arg1: i8, %arg2: i8, %arg3: i32, %arg4: i1, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i8) -> i1 attributes {llvm.emit_c_interface} {
    %0 = llvm.call @nan_caller(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8) : (i64, i8, i8, i32, i1, i1, i1, i1, i8) -> i1
    llvm.return %0 : i1
  }
  llvm.func @cast_caller(%arg0: i64, %arg1: i8, %arg2: i8, %arg3: i32, %arg4: i1, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i8, %arg9: i8, %arg10: i8, %arg11: i32, %arg12: i1, %arg13: i1, %arg14: i1, %arg15: i1, %arg16: i8) -> i64 attributes {llvm.emit_c_interface} {
    %0 = llvm.call @__float_cast(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16) : (i64, i8, i8, i32, i1, i1, i1, i1, i8, i8, i8, i32, i1, i1, i1, i1, i8) -> i64
    llvm.return %0 : i64
  }
  llvm.func @_mlir_ciface_cast_caller(%arg0: i64, %arg1: i8, %arg2: i8, %arg3: i32, %arg4: i1, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i8, %arg9: i8, %arg10: i8, %arg11: i32, %arg12: i1, %arg13: i1, %arg14: i1, %arg15: i1, %arg16: i8) -> i64 attributes {llvm.emit_c_interface} {
    %0 = llvm.call @cast_caller(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16) : (i64, i8, i8, i32, i1, i1, i1, i1, i8, i8, i8, i32, i1, i1, i1, i1, i8) -> i64
    llvm.return %0 : i64
  }
  llvm.func @castfloat_caller(%arg0: f64, %arg1: i8, %arg2: i8, %arg3: i32, %arg4: i1, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i8) -> i64 attributes {llvm.emit_c_interface} {
    %0 = llvm.bitcast %arg0 : f64 to i64
    %1 = llvm.mlir.constant(11 : i8) : i8
    %2 = llvm.mlir.constant(52 : i8) : i8
    %3 = llvm.mlir.constant(-1023 : i32) : i32
    %4 = llvm.mlir.constant(true) : i1
    %5 = llvm.mlir.constant(true) : i1
    %6 = llvm.mlir.constant(true) : i1
    %7 = llvm.mlir.constant(true) : i1
    %8 = llvm.mlir.constant(-1 : i8) : i8
    %9 = llvm.call @__float_cast(%0, %1, %2, %3, %4, %5, %6, %7, %8, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8) : (i64, i8, i8, i32, i1, i1, i1, i1, i8, i8, i8, i32, i1, i1, i1, i1, i8) -> i64
    llvm.return %9 : i64
  }
  llvm.func @_mlir_ciface_castfloat_caller(%arg0: f64, %arg1: i8, %arg2: i8, %arg3: i32, %arg4: i1, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i8) -> i64 attributes {llvm.emit_c_interface} {
    %0 = llvm.call @castfloat_caller(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8) : (f64, i8, i8, i32, i1, i1, i1, i1, i8) -> i64
    llvm.return %0 : i64
  }
  llvm.func @casttofloat_caller(%arg0: i64, %arg1: i8, %arg2: i8, %arg3: i32, %arg4: i1, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i8) -> f64 attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(11 : i8) : i8
    %1 = llvm.mlir.constant(52 : i8) : i8
    %2 = llvm.mlir.constant(-1023 : i32) : i32
    %3 = llvm.mlir.constant(true) : i1
    %4 = llvm.mlir.constant(true) : i1
    %5 = llvm.mlir.constant(true) : i1
    %6 = llvm.mlir.constant(true) : i1
    %7 = llvm.mlir.constant(-1 : i8) : i8
    %8 = llvm.call @__float_cast(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %0, %1, %2, %3, %4, %5, %6, %7) : (i64, i8, i8, i32, i1, i1, i1, i1, i8, i8, i8, i32, i1, i1, i1, i1, i8) -> i64
    %9 = llvm.bitcast %8 : i64 to f64
    llvm.return %9 : f64
  }
  llvm.func @_mlir_ciface_casttofloat_caller(%arg0: i64, %arg1: i8, %arg2: i8, %arg3: i32, %arg4: i1, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i8) -> f64 attributes {llvm.emit_c_interface} {
    %0 = llvm.call @casttofloat_caller(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8) : (i64, i8, i8, i32, i1, i1, i1, i1, i8) -> f64
    llvm.return %0 : f64
  }
}

