// RUN: base2-opt %s --canonicalize | FileCheck %s

//===----------------------------------------------------------------------===//
// value_cast
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @value_cast_ui_si_of(
func.func @value_cast_ui_si_of() -> (si8, si8, si8, si8, si8) {
    // CHECK-DAG: %[[MOD:.+]] = base2.constant -16 : si8
    // CHECK-DAG: %[[SAT:.+]] = base2.constant 127 : si8
    %cst = base2.constant 240 : ui8
    %0 = base2.value_cast %cst : ui8 to si8
    %1 = base2.value_cast %cst : ui8 to nearest si8
    %2 = base2.value_cast %cst : ui8 to round_down si8
    %3 = base2.value_cast %cst : ui8 to towards_zero si8
    %4 = base2.value_cast %cst : ui8 to away_from_zero si8
    // CHECK: return %[[MOD]], %[[SAT]], %[[SAT]], %[[MOD]], %[[SAT]]
    return %0, %1, %2, %3, %4 : si8, si8, si8, si8, si8
}

// CHECK-LABEL: func.func @value_cast_si_ui_uf(
func.func @value_cast_si_ui_uf() -> (ui8, ui8, ui8, ui8, ui8) {
    // CHECK-DAG: %[[MOD:.+]] = base2.constant 238 : ui8
    // CHECK-DAG: %[[SAT:.+]] = base2.constant 0 : ui8
    %cst = base2.constant -18 : si8
    %0 = base2.value_cast %cst : si8 to ui8
    %1 = base2.value_cast %cst : si8 to nearest ui8
    %2 = base2.value_cast %cst : si8 to round_up ui8
    %3 = base2.value_cast %cst : si8 to towards_zero ui8
    %4 = base2.value_cast %cst : si8 to away_from_zero ui8
    // CHECK: return %[[MOD]], %[[SAT]], %[[SAT]], %[[SAT]], %[[MOD]]
    return %0, %1, %2, %3, %4 : ui8, ui8, ui8, ui8, ui8
}

// CHECK-LABEL: func.func @value_cast_ui_ui_of(
func.func @value_cast_ui_ui_of() -> (ui4, ui4, ui4, ui4, ui4) {
    // CHECK-DAG: %[[MOD:.+]] = base2.constant 4 : ui4
    // CHECK-DAG: %[[SAT:.+]] = base2.constant 15 : ui4
    %cst = base2.constant 68 : ui8
    %0 = base2.value_cast %cst : ui8 to ui4
    %1 = base2.value_cast %cst : ui8 to nearest ui4
    %2 = base2.value_cast %cst : ui8 to round_down ui4
    %3 = base2.value_cast %cst : ui8 to towards_zero ui4
    %4 = base2.value_cast %cst : ui8 to away_from_zero ui4
    // CHECK: return %[[MOD]], %[[SAT]], %[[SAT]], %[[MOD]], %[[SAT]]
    return %0, %1, %2, %3, %4 : ui4, ui4, ui4, ui4, ui4
}

// CHECK-LABEL: func.func @value_cast_si_si_of(
func.func @value_cast_si_si_of() -> (si4, si4, si4, si4, si4) {
    // CHECK-DAG: %[[MOD:.+]] = base2.constant -4 : si4
    // CHECK-DAG: %[[SAT:.+]] = base2.constant 7 : si4
    %cst = base2.constant 76 : si8
    %0 = base2.value_cast %cst : si8 to si4
    %1 = base2.value_cast %cst : si8 to nearest si4
    %2 = base2.value_cast %cst : si8 to round_down si4
    %3 = base2.value_cast %cst : si8 to towards_zero si4
    %4 = base2.value_cast %cst : si8 to away_from_zero si4
    // CHECK: return %[[MOD]], %[[SAT]], %[[SAT]], %[[MOD]], %[[SAT]]
    return %0, %1, %2, %3, %4 : si4, si4, si4, si4, si4
}

// CHECK-LABEL: func.func @value_cast_si_si_uf(
func.func @value_cast_si_si_uf() -> (si4, si4, si4, si4, si4) {
    // CHECK-DAG: %[[MOD:.+]] = base2.constant -2 : si4
    // CHECK-DAG: %[[SAT:.+]] = base2.constant -8 : si4
    %cst = base2.constant -18 : si8
    %0 = base2.value_cast %cst : si8 to si4
    %1 = base2.value_cast %cst : si8 to nearest si4
    %2 = base2.value_cast %cst : si8 to round_up si4
    %3 = base2.value_cast %cst : si8 to towards_zero si4
    %4 = base2.value_cast %cst : si8 to away_from_zero si4
    // CHECK: return %[[MOD]], %[[SAT]], %[[SAT]], %[[SAT]], %[[MOD]]
    return %0, %1, %2, %3, %4 : si4, si4, si4, si4, si4
}

//===----------------------------------------------------------------------===//
// cmp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @cmp_ui(
func.func @cmp_ui() -> (i1, i1, i1, i1, i1, i1, i1) {
    // CHECK-DAG: %[[FALSE:.+]] = base2.constant false
    // CHECK-DAG: %[[TRUE:.+]] = base2.constant true
    %lhs = base2.constant 4 : ui8
    %rhs = base2.constant 8 : ui8
    %0 = base2.cmp oeq %lhs, %rhs : ui8
    %1 = base2.cmp ogt %lhs, %rhs : ui8
    %2 = base2.cmp oge %lhs, %rhs : ui8
    %3 = base2.cmp olt %lhs, %rhs : ui8
    %4 = base2.cmp ole %lhs, %rhs : ui8
    %5 = base2.cmp one %lhs, %rhs : ui8
    %6 = base2.cmp ord %lhs, %rhs : ui8
    // return %[[FALSE]], %[[FALSE]], %[[FALSE]], %[[TRUE]], %[[TRUE]], %[[TRUE]], %[[TRUE]]
    return %0, %1, %2, %3, %4, %5, %6 : i1, i1, i1, i1, i1, i1, i1
}

// CHECK-LABEL: func.func @cmp_si(
func.func @cmp_si() -> (i1, i1, i1, i1, i1, i1, i1) {
    // CHECK-DAG: %[[FALSE:.+]] = base2.constant false
    // CHECK-DAG: %[[TRUE:.+]] = base2.constant true
    %lhs = base2.constant -8 : si8
    %rhs = base2.constant 4 : si8
    %0 = base2.cmp oeq %lhs, %rhs : si8
    %1 = base2.cmp ogt %lhs, %rhs : si8
    %2 = base2.cmp oge %lhs, %rhs : si8
    %3 = base2.cmp olt %lhs, %rhs : si8
    %4 = base2.cmp ole %lhs, %rhs : si8
    %5 = base2.cmp one %lhs, %rhs : si8
    %6 = base2.cmp ord %lhs, %rhs : si8
    // return %[[FALSE]], %[[FALSE]], %[[FALSE]], %[[TRUE]], %[[TRUE]], %[[TRUE]], %[[TRUE]]
    return %0, %1, %2, %3, %4, %5, %6 : i1, i1, i1, i1, i1, i1, i1
}

//===----------------------------------------------------------------------===//
// add
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @add_ui_of(
func.func @add_ui_of() -> (ui4, ui4, ui4, ui4, ui4) {
    // CHECK-DAG: %[[MOD:.+]] = base2.constant 1 : ui4
    // CHECK-DAG: %[[SAT:.+]] = base2.constant 15 : ui4
    %lhs = base2.constant 7 : ui4
    %rhs = base2.constant 10 : ui4
    %0 = base2.add %lhs, %rhs : ui4
    %1 = base2.add %lhs, %rhs : nearest ui4
    %2 = base2.add %lhs, %rhs : round_down ui4
    %3 = base2.add %lhs, %rhs : towards_zero ui4
    %4 = base2.add %lhs, %rhs : away_from_zero ui4
    // CHECK: return %[[MOD]], %[[SAT]], %[[SAT]], %[[MOD]], %[[SAT]]
    return %0, %1, %2, %3, %4 : ui4, ui4, ui4, ui4, ui4
}

// CHECK-LABEL: func.func @add_si_of(
func.func @add_si_of() -> (si4, si4, si4, si4, si4) {
    // CHECK-DAG: %[[MOD:.+]] = base2.constant -5 : si4
    // CHECK-DAG: %[[SAT:.+]] = base2.constant 7 : si4
    %lhs = base2.constant 7 : si4
    %rhs = base2.constant 4 : si4
    %0 = base2.add %lhs, %rhs : si4
    %1 = base2.add %lhs, %rhs : nearest si4
    %2 = base2.add %lhs, %rhs : round_down si4
    %3 = base2.add %lhs, %rhs : towards_zero si4
    %4 = base2.add %lhs, %rhs : away_from_zero si4
    // CHECK: return %[[MOD]], %[[SAT]], %[[SAT]], %[[MOD]], %[[SAT]]
    return %0, %1, %2, %3, %4 : si4, si4, si4, si4, si4
}

// CHECK-LABEL: func.func @add_si_uf(
func.func @add_si_uf() -> (si4, si4, si4, si4, si4) {
    // CHECK-DAG: %[[MOD:.+]] = base2.constant 7 : si4
    // CHECK-DAG: %[[SAT:.+]] = base2.constant -8 : si4
    %lhs = base2.constant -4 : si4
    %rhs = base2.constant -5 : si4
    %0 = base2.add %lhs, %rhs : si4
    %1 = base2.add %lhs, %rhs : nearest si4
    %2 = base2.add %lhs, %rhs : round_up si4
    %3 = base2.add %lhs, %rhs : towards_zero si4
    %4 = base2.add %lhs, %rhs : away_from_zero si4
    // CHECK: return %[[MOD]], %[[SAT]], %[[SAT]], %[[SAT]], %[[MOD]]
    return %0, %1, %2, %3, %4 : si4, si4, si4, si4, si4
}

//===----------------------------------------------------------------------===//
// sub
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @sub_ui_uf(
func.func @sub_ui_uf() -> (ui4, ui4, ui4, ui4, ui4) {
    // CHECK-DAG: %[[MOD:.+]] = base2.constant 13 : ui4
    // CHECK-DAG: %[[SAT:.+]] = base2.constant 0 : ui4
    %lhs = base2.constant 3 : ui4
    %rhs = base2.constant 6 : ui4
    %0 = base2.sub %lhs, %rhs : ui4
    %1 = base2.sub %lhs, %rhs : nearest ui4
    %2 = base2.sub %lhs, %rhs : round_up ui4
    %3 = base2.sub %lhs, %rhs : towards_zero ui4
    %4 = base2.sub %lhs, %rhs : away_from_zero ui4
    // CHECK: return %[[MOD]], %[[SAT]], %[[SAT]], %[[SAT]], %[[MOD]]
    return %0, %1, %2, %3, %4 : ui4, ui4, ui4, ui4, ui4
}

// CHECK-LABEL: func.func @sub_si_of(
func.func @sub_si_of() -> (si4, si4, si4, si4, si4) {
    // CHECK-DAG: %[[MOD:.+]] = base2.constant -5 : si4
    // CHECK-DAG: %[[SAT:.+]] = base2.constant 7 : si4
    %lhs = base2.constant 7 : si4
    %rhs = base2.constant -4 : si4
    %0 = base2.sub %lhs, %rhs : si4
    %1 = base2.sub %lhs, %rhs : nearest si4
    %2 = base2.sub %lhs, %rhs : round_down si4
    %3 = base2.sub %lhs, %rhs : towards_zero si4
    %4 = base2.sub %lhs, %rhs : away_from_zero si4
    // CHECK: return %[[MOD]], %[[SAT]], %[[SAT]], %[[MOD]], %[[SAT]]
    return %0, %1, %2, %3, %4 : si4, si4, si4, si4, si4
}

// CHECK-LABEL: func.func @sub_si_uf(
func.func @sub_si_uf() -> (si4, si4, si4, si4, si4) {
    // CHECK-DAG: %[[MOD:.+]] = base2.constant 7 : si4
    // CHECK-DAG: %[[SAT:.+]] = base2.constant -8 : si4
    %lhs = base2.constant -3 : si4
    %rhs = base2.constant 6 : si4
    %0 = base2.sub %lhs, %rhs : si4
    %1 = base2.sub %lhs, %rhs : nearest si4
    %2 = base2.sub %lhs, %rhs : round_up si4
    %3 = base2.sub %lhs, %rhs : towards_zero si4
    %4 = base2.sub %lhs, %rhs : away_from_zero si4
    // CHECK: return %[[MOD]], %[[SAT]], %[[SAT]], %[[SAT]], %[[MOD]]
    return %0, %1, %2, %3, %4 : si4, si4, si4, si4, si4
}

//===----------------------------------------------------------------------===//
// mul
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @mul_ui_of(
func.func @mul_ui_of() -> (ui4, ui4, ui4, ui4, ui4) {
    // CHECK-DAG: %[[MOD:.+]] = base2.constant 2 : ui4
    // CHECK-DAG: %[[SAT:.+]] = base2.constant 15 : ui4
    %lhs = base2.constant 3 : ui4
    %rhs = base2.constant 6 : ui4
    %0 = base2.mul %lhs, %rhs : ui4
    %1 = base2.mul %lhs, %rhs : nearest ui4
    %2 = base2.mul %lhs, %rhs : round_down ui4
    %3 = base2.mul %lhs, %rhs : towards_zero ui4
    %4 = base2.mul %lhs, %rhs : away_from_zero ui4
    // CHECK: return %[[MOD]], %[[SAT]], %[[SAT]], %[[MOD]], %[[SAT]]
    return %0, %1, %2, %3, %4 : ui4, ui4, ui4, ui4, ui4
}

// CHECK-LABEL: func.func @mul_si_of(
func.func @mul_si_of() -> (si4, si4, si4, si4, si4) {
    // CHECK-DAG: %[[MOD:.+]] = base2.constant -7 : si4
    // CHECK-DAG: %[[SAT:.+]] = base2.constant 7 : si4
    %lhs = base2.constant -3 : si4
    %rhs = base2.constant -3 : si4
    %0 = base2.mul %lhs, %rhs : si4
    %1 = base2.mul %lhs, %rhs : nearest si4
    %2 = base2.mul %lhs, %rhs : round_down si4
    %3 = base2.mul %lhs, %rhs : towards_zero si4
    %4 = base2.mul %lhs, %rhs : away_from_zero si4
    // CHECK: return %[[MOD]], %[[SAT]], %[[SAT]], %[[MOD]], %[[SAT]]
    return %0, %1, %2, %3, %4 : si4, si4, si4, si4, si4
}

//===----------------------------------------------------------------------===//
// div
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @div_si_of(
func.func @div_si_of() -> (si4, si4, si4, si4, si4) {
    // CHECK-DAG: %[[MOD:.+]] = base2.constant -8 : si4
    // CHECK-DAG: %[[SAT:.+]] = base2.constant 7 : si4
    %lhs = base2.constant -8 : si4
    %rhs = base2.constant -1 : si4
    %0 = base2.div %lhs, %rhs : si4
    %1 = base2.div %lhs, %rhs : nearest si4
    %2 = base2.div %lhs, %rhs : round_down si4
    %3 = base2.div %lhs, %rhs : towards_zero si4
    %4 = base2.div %lhs, %rhs : away_from_zero si4
    // CHECK: return %[[MOD]], %[[SAT]], %[[SAT]], %[[MOD]], %[[SAT]]
    return %0, %1, %2, %3, %4 : si4, si4, si4, si4, si4
}

//===----------------------------------------------------------------------===//
// mod
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @mod_si_sign(
func.func @mod_si_sign() -> (si8, si8, si8, si8) {
    // CHECK-DAG: %[[REM_POS:.+]] = base2.constant 1 : si8
    // CHECK-DAG: %[[REM_NEG:.+]] = base2.constant -1 : si8
    %lhs_pos = base2.constant 10 : si8
    %lhs_neg = base2.constant -10 : si8
    %rhs_pos = base2.constant 3 : si8
    %rhs_neg = base2.constant -3 : si8
    %0 = base2.mod %lhs_pos, %rhs_pos : si8
    %1 = base2.mod %lhs_pos, %rhs_neg : si8
    %2 = base2.mod %lhs_neg, %rhs_pos : si8
    %3 = base2.mod %lhs_neg, %rhs_neg : si8
    // CHECK: return %[[REM_POS]], %[[REM_POS]], %[[REM_NEG]], %[[REM_NEG]]
    return %0, %1, %2, %3 : si8, si8, si8, si8
}

// CHECK-LABEL: func.func @mod_si_of(
func.func @mod_si_of() -> (si4) {
    // CHECK-DAG: %[[REM:.+]] = base2.constant 0 : si4
    %lhs = base2.constant -8 : si4
    %rhs = base2.constant -1 : si4
    %0 = base2.mod %lhs, %rhs : si4
    // CHECK: return %[[REM]]
    return %0 : si4
}
