; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

declare i64 @__float_cast(i64, i8, i8, i32, i1, i1, i1, i1, i8, i8, i8, i32, i1, i1, i1, i1, i8)

declare i1 @__float64_is_signaling_nan(i64, i8, i8, i32, i1, i1, i1, i1, i8)

declare i1 @__float_ltgt_quiet(i64, i64, i8, i8, i32, i1, i1, i1, i1, i8)

declare i1 @__float_gt(i64, i64, i8, i8, i32, i1, i1, i1, i1, i8)

declare i1 @__float_ge(i64, i64, i8, i8, i32, i1, i1, i1, i1, i8)

declare i1 @__float_lt(i64, i64, i8, i8, i32, i1, i1, i1, i1, i8)

declare i1 @__float_le(i64, i64, i8, i8, i32, i1, i1, i1, i1, i8)

declare i1 @__float_eq(i64, i64, i8, i8, i32, i1, i1, i1, i1, i8)

declare i64 @__float_divSRT4(i64, i64, i8, i8, i32, i1, i1, i1, i1, i8)

declare i64 @__float_divG(i64, i64, i8, i8, i32, i1, i1, i1, i1, i8)

declare i64 @__float_mul(i64, i64, i8, i8, i32, i1, i1, i1, i1, i8)

declare i64 @__float_sub(i64, i64, i8, i8, i32, i1, i1, i1, i1, i8)

declare i64 @__float_add(i64, i64, i8, i8, i32, i1, i1, i1, i1, i8)

define i64 @add_caller(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9) {
  %11 = call i64 @__float_add(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9)
  ret i64 %11
}

define i64 @_mlir_ciface_add_caller(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9) {
  %11 = call i64 @add_caller(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9)
  ret i64 %11
}

define i64 @sub_caller(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9) {
  %11 = call i64 @__float_sub(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9)
  ret i64 %11
}

define i64 @_mlir_ciface_sub_caller(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9) {
  %11 = call i64 @sub_caller(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9)
  ret i64 %11
}

define i64 @mul_caller(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9) {
  %11 = call i64 @__float_mul(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9)
  ret i64 %11
}

define i64 @_mlir_ciface_mul_caller(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9) {
  %11 = call i64 @mul_caller(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9)
  ret i64 %11
}

define i64 @divg_caller(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9) {
  %11 = call i64 @__float_divG(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9)
  ret i64 %11
}

define i64 @_mlir_ciface_divg_caller(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9) {
  %11 = call i64 @divg_caller(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9)
  ret i64 %11
}

define i64 @divsrt_caller(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9) {
  %11 = call i64 @__float_divSRT4(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9)
  ret i64 %11
}

define i64 @_mlir_ciface_divsrt_caller(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9) {
  %11 = call i64 @divsrt_caller(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9)
  ret i64 %11
}

define i1 @eq_caller(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9) {
  %11 = call i1 @__float_eq(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9)
  ret i1 %11
}

define i1 @_mlir_ciface_eq_caller(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9) {
  %11 = call i1 @eq_caller(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9)
  ret i1 %11
}

define i1 @le_caller(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9) {
  %11 = call i1 @__float_le(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9)
  ret i1 %11
}

define i1 @_mlir_ciface_le_caller(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9) {
  %11 = call i1 @le_caller(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9)
  ret i1 %11
}

define i1 @lt_caller(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9) {
  %11 = call i1 @__float_lt(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9)
  ret i1 %11
}

define i1 @_mlir_ciface_lt_caller(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9) {
  %11 = call i1 @lt_caller(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9)
  ret i1 %11
}

define i1 @ge_caller(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9) {
  %11 = call i1 @__float_ge(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9)
  ret i1 %11
}

define i1 @_mlir_ciface_ge_caller(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9) {
  %11 = call i1 @ge_caller(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9)
  ret i1 %11
}

define i1 @gt_caller(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9) {
  %11 = call i1 @__float_gt(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9)
  ret i1 %11
}

define i1 @_mlir_ciface_gt_caller(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9) {
  %11 = call i1 @gt_caller(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9)
  ret i1 %11
}

define i1 @ltgt_caller(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9) {
  %11 = call i1 @__float_ltgt_quiet(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9)
  ret i1 %11
}

define i1 @_mlir_ciface_ltgt_caller(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9) {
  %11 = call i1 @ltgt_caller(i64 %0, i64 %1, i8 %2, i8 %3, i32 %4, i1 %5, i1 %6, i1 %7, i1 %8, i8 %9)
  ret i1 %11
}

define i1 @nan_caller(i64 %0, i8 %1, i8 %2, i32 %3, i1 %4, i1 %5, i1 %6, i1 %7, i8 %8) {
  %10 = call i1 @__float64_is_signaling_nan(i64 %0, i8 %1, i8 %2, i32 %3, i1 %4, i1 %5, i1 %6, i1 %7, i8 %8)
  ret i1 %10
}

define i1 @_mlir_ciface_nan_caller(i64 %0, i8 %1, i8 %2, i32 %3, i1 %4, i1 %5, i1 %6, i1 %7, i8 %8) {
  %10 = call i1 @nan_caller(i64 %0, i8 %1, i8 %2, i32 %3, i1 %4, i1 %5, i1 %6, i1 %7, i8 %8)
  ret i1 %10
}

define i64 @cast_caller(i64 %0, i8 %1, i8 %2, i32 %3, i1 %4, i1 %5, i1 %6, i1 %7, i8 %8, i8 %9, i8 %10, i32 %11, i1 %12, i1 %13, i1 %14, i1 %15, i8 %16) {
  %18 = call i64 @__float_cast(i64 %0, i8 %1, i8 %2, i32 %3, i1 %4, i1 %5, i1 %6, i1 %7, i8 %8, i8 %9, i8 %10, i32 %11, i1 %12, i1 %13, i1 %14, i1 %15, i8 %16)
  ret i64 %18
}

define i64 @_mlir_ciface_cast_caller(i64 %0, i8 %1, i8 %2, i32 %3, i1 %4, i1 %5, i1 %6, i1 %7, i8 %8, i8 %9, i8 %10, i32 %11, i1 %12, i1 %13, i1 %14, i1 %15, i8 %16) {
  %18 = call i64 @cast_caller(i64 %0, i8 %1, i8 %2, i32 %3, i1 %4, i1 %5, i1 %6, i1 %7, i8 %8, i8 %9, i8 %10, i32 %11, i1 %12, i1 %13, i1 %14, i1 %15, i8 %16)
  ret i64 %18
}

define i64 @castfloat_caller(double %0, i8 %1, i8 %2, i32 %3, i1 %4, i1 %5, i1 %6, i1 %7, i8 %8) {
  %10 = bitcast double %0 to i64
  %11 = call i64 @__float_cast(i64 %10, i8 11, i8 52, i32 -1023, i1 true, i1 true, i1 true, i1 true, i8 -1, i8 %1, i8 %2, i32 %3, i1 %4, i1 %5, i1 %6, i1 %7, i8 %8)
  ret i64 %11
}

define i64 @_mlir_ciface_castfloat_caller(double %0, i8 %1, i8 %2, i32 %3, i1 %4, i1 %5, i1 %6, i1 %7, i8 %8) {
  %10 = call i64 @castfloat_caller(double %0, i8 %1, i8 %2, i32 %3, i1 %4, i1 %5, i1 %6, i1 %7, i8 %8)
  ret i64 %10
}

define double @casttofloat_caller(i64 %0, i8 %1, i8 %2, i32 %3, i1 %4, i1 %5, i1 %6, i1 %7, i8 %8) {
  %10 = call i64 @__float_cast(i64 %0, i8 %1, i8 %2, i32 %3, i1 %4, i1 %5, i1 %6, i1 %7, i8 %8, i8 11, i8 52, i32 -1023, i1 true, i1 true, i1 true, i1 true, i8 -1)
  %11 = bitcast i64 %10 to double
  ret double %11
}

define double @_mlir_ciface_casttofloat_caller(i64 %0, i8 %1, i8 %2, i32 %3, i1 %4, i1 %5, i1 %6, i1 %7, i8 %8) {
  %10 = call double @casttofloat_caller(i64 %0, i8 %1, i8 %2, i32 %3, i1 %4, i1 %5, i1 %6, i1 %7, i8 %8)
  ret double %10
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
