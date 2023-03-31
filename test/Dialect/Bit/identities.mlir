// RUN: base2-opt --canonicalize --cse %s | FileCheck %s

//===----------------------------------------------------------------------===//
// cast
//===----------------------------------------------------------------------===//

// cast x : T to T = x
// CHECK-LABEL: func.func @cast_noop(
// CHECK-SAME: %[[ARG0:.+]]: i64
func.func @cast_noop(%arg0: i64) -> i64 {
    %0 = bit.cast %arg0 : i64 to i64
    // CHECK: return %[[ARG0]]
    return %0 : i64
}

// cast (cast x : T to U) : U to V = cast x : T to V
// CHECK-LABEL: func.func @cast_transitive(
// CHECK-SAME: %[[ARG0:.+]]: i64
func.func @cast_transitive(%arg0: i64) -> si64 {
    // CHECK-DAG: %[[CAST0:.+]] = bit.cast %[[ARG0]] : i64 to si64
    %0 = bit.cast %arg0 : i64 to ui64
    %1 = bit.cast %0 : ui64 to si64
    // CHECK: return %[[CAST0]]
    return %1 : si64
}

// cast poison : T to U = poison
// CHECK-LABEL: func.func @cast_poison(
func.func @cast_poison() -> si64 {
    // CHECK-DAG: %[[POISON:.+]] = ub.poison : si64
    %poison = ub.poison : i64
    %0 = bit.cast %poison : i64 to si64
    // CHECK: return %[[POISON]]
    return %0 : si64
}

//===----------------------------------------------------------------------===//
// cmp
//===----------------------------------------------------------------------===//

// cmp P x, y = cmp P y, x
// CHECK-LABEL: func.func @cmp_commutative(
// CHECK-SAME: %[[ARG0:.+]]: i64,
// CHECK-SAME: %[[ARG1:.+]]: i64
func.func @cmp_commutative(%arg0: i64, %arg1: i64)
        -> (i1, i1, i1, i1, i1, i1, i1, i1) {
    // CHECK-DAG: %[[CST1:.+]] = bit.constant 1 : i64
    %cst1 = bit.constant 1 : i64
    %0 = bit.cmp eq %arg0, %arg1 : i64
    %1 = bit.cmp eq %arg1, %arg0 : i64
    // CHECK-DAG: %[[ARGEQ:.+]] = bit.cmp eq %[[ARG0]], %[[ARG1]]
    %2 = bit.cmp eq %arg0, %cst1 : i64
    %3 = bit.cmp eq %cst1, %arg0 : i64
    // CHECK-DAG: %[[CSTEQ:.+]] = bit.cmp eq %[[ARG0]], %[[CST1]]
    %4 = bit.cmp ne %arg0, %arg1 : i64
    %5 = bit.cmp ne %arg1, %arg0 : i64
    // CHECK-DAG: %[[ARGNE:.+]] = bit.cmp ne %[[ARG0]], %[[ARG1]]
    %6 = bit.cmp ne %arg0, %cst1 : i64
    %7 = bit.cmp ne %cst1, %arg0 : i64
    // CHECK-DAG: %[[CSTNE:.+]] = bit.cmp ne %[[ARG0]], %[[CST1]]
    // CHECK: return %[[ARGEQ]], %[[ARGEQ]], %[[CSTEQ]], %[[CSTEQ]], %[[ARGNE]], %[[ARGNE]], %[[CSTNE]], %[[CSTNE]]
    return %0, %1, %2, %3, %4, %5, %6, %7 : i1, i1, i1, i1, i1, i1, i1, i1
}

// cmp P x, x = cmp P 1, 1
// CHECK-LABEL: func.func @cmp_reflexive(
func.func @cmp_reflexive(%arg0: i64) -> (i1, i1) {
    // CHECK-DAG: %[[TRUE:.+]] = bit.constant true
    // CHECK-DAG: %[[FALSE:.+]] = bit.constant false
    %0 = bit.cmp eq %arg0, %arg0 : i64
    %1 = bit.cmp ne %arg0, %arg0 : i64
    // CHECK: return %[[TRUE]], %[[FALSE]]
    return %0, %1 : i1, i1
}

// cmp P ?, poison = poison
// CHECK-LABEL: func.func @cmp_poison(
func.func @cmp_poison(%arg0: i64) -> (i1, i1, i1, i1) {
    // CHECK-DAG: %[[POISON:.+]] = ub.poison : i1
    %poison = ub.poison : i64
    %0 = bit.cmp eq %arg0, %poison : i64
    %1 = bit.cmp eq %poison, %poison : i64
    %2 = bit.cmp ne %arg0, %poison : i64
    %3 = bit.cmp ne %poison, %poison : i64
    // CHECK: return %[[POISON]], %[[POISON]], %[[POISON]], %[[POISON]]
    return %0, %1, %2, %3 : i1, i1, i1, i1
}

//===----------------------------------------------------------------------===//
// select
//===----------------------------------------------------------------------===//

// select x, y, y        = y
// select poison, x, x   = x
// select x, true, false = x
// CHECK-LABEL: func.func @select_noop(
// CHECK-SAME: %[[ARG0:.+]]: i1,
// CHECK-SAME: %[[ARG1:.+]]: i64
func.func @select_noop(%arg0: i1, %arg1: i64) -> (i64, i64, i1) {
    %poison = ub.poison : i1
    %true = bit.constant true
    %false = bit.constant false
    %0 = bit.select %arg0, %arg1, %arg1 : i64
    %1 = bit.select %poison, %arg1, %arg1 : i64
    %2 = bit.select %arg0, %true, %false : i1
    // return %[[ARG1]], %[[ARG1]], %[[ARG0]]
    return %0, %1, %2 : i64, i64, i1
}

// select poison, x, y     = poison
// select true, poison, ?  = poison
// select false, ?, poison = poison
// CHECK-LABEL: func.func @select_poison(
func.func @select_poison(%arg0: i64, %arg1: i64) -> (i64, i64, i64) {
    // CHECK-DAG: %[[POISON:.+]] = ub.poison : i64
    %poison_i1 = ub.poison : i1
    %poison_i64 = ub.poison : i64
    %true = bit.constant true
    %false = bit.constant false
    %0 = bit.select %poison_i1, %arg0, %arg1 : i64
    %1 = bit.select %true, %poison_i64, %arg1 : i64
    %2 = bit.select %false, %arg0, %poison_i64 : i64
    // RETURN: %[[POISON]], %[[POISON]], %[[POISON]]
    return %0, %1, %2 : i64, i64, i64
}

//===----------------------------------------------------------------------===//
// and
//===----------------------------------------------------------------------===//

// and x, y = and y, x
// CHECK-LABEL: func.func @and_commutative(
// CHECK-SAME: %[[ARG0:.+]]: i64,
// CHECK-SAME: %[[ARG1:.+]]: i64
func.func @and_commutative(%arg0: i64, %arg1: i64) -> (i64, i64, i64, i64) {
    // CHECK-DAG: %[[CST1:.+]] = bit.constant 1 : i64
    %cst1 = bit.constant 1 : i64
    %0 = bit.and %arg0, %arg1 : i64
    %1 = bit.and %arg1, %arg0 : i64
    // CHECK-DAG: %[[ARG:.+]] = bit.and %[[ARG0]], %[[ARG1]]
    %2 = bit.and %arg0, %cst1 : i64
    %3 = bit.and %cst1, %arg0 : i64
    // CHECK-DAG: %[[CST:.+]] = bit.and %[[ARG0]], %[[CST1]]
    // CHECK: return %[[ARG]], %[[ARG]], %[[CST]], %[[CST]]
    return %0, %1, %2, %3 : i64, i64, i64, i64
}

// and x, x = x
// CHECK-LABEL: func.func @and_noop(
// CHECK-SAME: %[[ARG0:.+]]: i64
func.func @and_noop(%arg0: i64) -> i64 {
    %0 = bit.and %arg0, %arg0 : i64
    // return %[[ARG0]]
    return %0 : i64
}

// and poison, poison = poison
// CHECK-LABEL: func.func @and_poison(
// CHECK-SAME: %[[ARG0:.+]]: i64
func.func @and_poison(%arg0: i64) -> (i64, i64) {
    // CHECK-DAG: %[[POISON:.+]] = ub.poison : i64
    %poison = ub.poison : i64
    // CHECK-DAG: %[[AND:.+]] = bit.and %[[ARG0]], %[[POISON]]
    %0 = bit.and %arg0, %poison : i64
    %1 = bit.and %poison, %poison : i64
    // return %[[AND]], %[[POISON]]
    return %0, %1 : i64, i64
}

// and ?, 0 = 0
// CHECK-LABEL: func.func @and_min(
func.func @and_min(%arg0: i64) -> (i64, i64) {
    // CHECK-DAG: %[[CST0:.+]] = bit.constant 0 : i64
    %cst0 = bit.constant 0 : i64
    %poison = ub.poison : i64
    %0 = bit.and %arg0, %cst0 : i64
    %1 = bit.and %poison, %cst0 : i64
    // CHECK: return %[[CST0]], %[[CST0]]
    return %0, %1 : i64, i64
}

// and x, 2^N-1 = x
// CHECK-LABEL: func.func @and_max(
// CHECK-SAME: %[[ARG0:.+]]: i64
func.func @and_max(%arg0: i64) -> (i64, i64) {
    // CHECK-DAG: %[[POISON:.+]] = ub.poison : i64
    %cst2Nm1 = bit.constant -1 : i64
    %poison = ub.poison : i64
    %0 = bit.and %arg0, %cst2Nm1 : i64
    %1 = bit.and %poison, %cst2Nm1 : i64
    // CHECK: return %[[ARG0]], %[[POISON]]
    return %0, %1 : i64, i64
}

//===----------------------------------------------------------------------===//
// or
//===----------------------------------------------------------------------===//

// or x, y = or y, x
// CHECK-LABEL: func.func @or_commutative(
// CHECK-SAME: %[[ARG0:.+]]: i64,
// CHECK-SAME: %[[ARG1:.+]]: i64
func.func @or_commutative(%arg0: i64, %arg1: i64) -> (i64, i64, i64, i64) {
    // CHECK-DAG: %[[CST1:.+]] = bit.constant 1 : i64
    %cst1 = bit.constant 1 : i64
    %0 = bit.or %arg0, %arg1 : i64
    %1 = bit.or %arg1, %arg0 : i64
    // CHECK-DAG: %[[ARG:.+]] = bit.or %[[ARG0]], %[[ARG1]]
    %2 = bit.or %arg0, %cst1 : i64
    %3 = bit.or %cst1, %arg0 : i64
    // CHECK-DAG: %[[CST:.+]] = bit.or %[[ARG0]], %[[CST1]]
    // CHECK: return %[[ARG]], %[[ARG]], %[[CST]], %[[CST]]
    return %0, %1, %2, %3 : i64, i64, i64, i64
}

// or x, x = x
// CHECK-LABEL: func.func @or_noop(
// CHECK-SAME: %[[ARG0:.+]]: i64
func.func @or_noop(%arg0: i64) -> i64 {
    %0 = bit.or %arg0, %arg0 : i64
    // return %[[ARG0]]
    return %0 : i64
}

// or poison, poison = poison
// CHECK-LABEL: func.func @or_poison(
// CHECK-SAME: %[[ARG0:.+]]: i64
func.func @or_poison(%arg0: i64) -> (i64, i64) {
    // CHECK-DAG: %[[POISON:.+]] = ub.poison : i64
    %poison = ub.poison : i64
    // CHECK-DAG: %[[OR:.+]] = bit.or %[[ARG0]], %[[POISON]]
    %0 = bit.or %arg0, %poison : i64
    %1 = bit.or %poison, %poison : i64
    // return %[[OR]], %[[POISON]]
    return %0, %1 : i64, i64
}

// or ?, 2^N-1 = 2^N-1
// CHECK-LABEL: func.func @or_max(
func.func @or_max(%arg0: i64) -> (i64, i64) {
    // CHECK-DAG: %[[CST2Nm1:.+]] = bit.constant -1 : i64
    %cst2Nm1 = bit.constant 0xFFFFFFFFFFFFFFFF : i64
    %poison = ub.poison : i64
    %0 = bit.or %arg0, %cst2Nm1 : i64
    %1 = bit.or %poison, %cst2Nm1 : i64
    // CHECK: return %[[CST2Nm1]], %[[CST2Nm1]]
    return %0, %1 : i64, i64
}

// or x, 0 = x
// CHECK-LABEL: func.func @or_min(
// CHECK-SAME: %[[ARG0:.+]]: i64
func.func @or_min(%arg0: i64) -> (i64, i64) {
    // CHECK-DAG: %[[POISON:.+]] = ub.poison : i64
    %poison = ub.poison : i64
    %cst0 = bit.constant 0 : i64
    %0 = bit.or %arg0, %cst0 : i64
    %1 = bit.or %poison, %cst0 : i64
    // CHECK: return %[[ARG0]], %[[POISON]]
    return %0, %1 : i64, i64
}

//===----------------------------------------------------------------------===//
// xor
//===----------------------------------------------------------------------===//

// xor x, y = xor y, x
// CHECK-LABEL: func.func @xor_commutative(
// CHECK-SAME: %[[ARG0:.+]]: i64,
// CHECK-SAME: %[[ARG1:.+]]: i64
func.func @xor_commutative(%arg0: i64, %arg1: i64) -> (i64, i64, i64, i64) {
    // CHECK-DAG: %[[CST1:.+]] = bit.constant 1 : i64
    %cst1 = bit.constant 1 : i64
    %0 = bit.xor %arg0, %arg1 : i64
    %1 = bit.xor %arg1, %arg0 : i64
    // CHECK-DAG: %[[ARG:.+]] = bit.xor %[[ARG0]], %[[ARG1]]
    %2 = bit.xor %arg0, %cst1 : i64
    %3 = bit.xor %cst1, %arg0 : i64
    // CHECK-DAG: %[[CST:.+]] = bit.xor %[[ARG0]], %[[CST1]]
    // CHECK: return %[[ARG]], %[[ARG]], %[[CST]], %[[CST]]
    return %0, %1, %2, %3 : i64, i64, i64, i64
}

// xor x, x = 0
// CHECK-LABEL: func.func @xor_zero(
// CHECK-SAME: %[[ARG0:.+]]: i64
func.func @xor_zero(%arg0: i64) -> i64 {
    // CHECK: %[[CST0:.+]] = bit.constant 0 : i64
    %0 = bit.xor %arg0, %arg0 : i64
    // return %[[CST0]]
    return %0 : i64
}

// xor poison, poison = poison
// CHECK-LABEL: func.func @xor_poison(
// CHECK-SAME: %[[ARG0:.+]]: i64
func.func @xor_poison(%arg0: i64) -> (i64, i64) {
    // CHECK-DAG: %[[POISON:.+]] = ub.poison : i64
    %poison = ub.poison : i64
    // CHECK-DAG: %[[XOR:.+]] = bit.xor %[[ARG0]], %[[POISON]]
    %0 = bit.xor %arg0, %poison : i64
    %1 = bit.xor %poison, %poison : i64
    // return %[[XOR]], %[[POISON]]
    return %0, %1 : i64, i64
}

// xor x, 0 = x
// CHECK-LABEL: func.func @xor_min(
// CHECK-SAME: %[[ARG0:.+]]: i64
func.func @xor_min(%arg0: i64) -> (i64, i64) {
    // CHECK-DAG: %[[POISON:.+]] = ub.poison : i64
    %poison = ub.poison : i64
    %cst0 = bit.constant 0 : i64
    %0 = bit.xor %arg0, %cst0 : i64
    %1 = bit.xor %poison, %cst0 : i64
    // CHECK: return %[[ARG0]], %[[POISON]]
    return %0, %1 : i64, i64
}

//===----------------------------------------------------------------------===//
// shl
//===----------------------------------------------------------------------===//

// shl x, 0   = x
// shl x:y, 0 = x
// CHECK-LABEL: func.func @shl_neutral(
// CHECK-SAME: %[[ARG0:.+]]: i64
func.func @shl_neutral(%arg0: i64, %arg1: i64) -> (i64, i64) {
    %cst0 = index.constant 0
    %0 = bit.shl %arg0, %cst0 : i64
    %1 = bit.shl %arg0:%arg1, %cst0 : i64
    // return %[[ARG0]], %[[ARG0]]
    return %0, %1 : i64, i64
}

// shl x, >=N         = 0
// shl poison, >=N    = 0
// shl x:y, >=2N      = 0
// shl ?:poison, >=2N = 0
// shl poison:?, >=2N = 0
// CHECK-LABEL: func.func @shl_zero(
func.func @shl_zero(%arg0: i64, %arg1: i64) -> (i64, i64, i64, i64, i64) {
    // CHECK-DAG: %[[CST0:.+]] = bit.constant 0 : i64
    %poison = ub.poison : i64
    %cst64 = index.constant 64
    %cst128 = index.constant 128
    %0 = bit.shl %arg0, %cst64 : i64
    %1 = bit.shl %poison, %cst64 : i64
    %2 = bit.shl %arg0:%arg1, %cst128 : i64
    %3 = bit.shl %arg0:%poison, %cst128 : i64
    %4 = bit.shl %poison:%arg1, %cst128 : i64
    // CHECK: return %[[CST0]], %[[CST0]], %[[CST0]], %[[CST0]], %[[CST0]]
    return %0, %1, %2, %3, %4 : i64, i64, i64, i64, i64
}

// shl poison, <N         = poison
// shl ?:poison, <2N      = poison
// shl poison:x, <N       = poison
// shl poison:poison, <2N = poison
// CHECK-LABEL: func.func @shl_poison(
// CHECK-SAME: %[[ARG0:.+]]: i64
func.func @shl_poison(%arg0: i64) -> (i64, i64, i64, i64, i64) {
    // CHECK-DAG: %[[POISON:.+]] = ub.poison : i64
    // CHECK-DAG: %[[CST63:.+]] = index.constant 63
    %poisoni64 = ub.poison : i64
    %cst63 = index.constant 63
    %cst127 = index.constant 127
    %0 = bit.shl %poisoni64, %cst63 : i64
    %1 = bit.shl %arg0:%poisoni64, %cst127 : i64
    // CHECK-DAG: %[[SHL:.+]] = bit.shl %[[ARG0]], %[[CST63]]
    %2 = bit.shl %poisoni64:%arg0, %cst127 : i64
    %3 = bit.shl %poisoni64:%arg0, %cst63 : i64
    %4 = bit.shl %poisoni64:%poisoni64, %cst127 : i64
    // CHECK: return %[[POISON]], %[[POISON]], %[[SHL]], %[[POISON]], %[[POISON]]
    return %0, %1, %2, %3, %4 : i64, i64, i64, i64, i64
}

// shl ?, poison   = poison
// shl ?:?, poison = poison
// CHECK-LABEL: func.func @shl_poison_amount(
func.func @shl_poison_amount(%arg0: i64, %arg1: i64)
        -> (i64, i64, i64, i64, i64) {
    // CHECK-DAG: %[[POISON:.+]] = ub.poison : i64
    %poisonidx = ub.poison : index
    %poisoni64 = ub.poison : i64
    %0 = bit.shl %poisoni64, %poisonidx : i64
    %1 = bit.shl %arg0:%arg1, %poisonidx : i64
    %2 = bit.shl %arg0:%poisoni64, %poisonidx : i64
    %3 = bit.shl %poisoni64:%arg0, %poisonidx : i64
    %4 = bit.shl %poisoni64:%poisoni64, %poisonidx : i64
    // CHECK: return %[[POISON]], %[[POISON]], %[[POISON]], %[[POISON]], %[[POISON]]
    return %0, %1, %2, %3, %4 : i64, i64, i64, i64, i64
}

//===----------------------------------------------------------------------===//
// shr
//===----------------------------------------------------------------------===//

// shr x, 0   = x
// shr x:y, 0 = x
// CHECK-LABEL: func.func @shr_neutral(
// CHECK-SAME: %[[ARG0:.+]]: i64
func.func @shr_neutral(%arg0: i64, %arg1: i64) -> (i64, i64) {
    %cst0 = index.constant 0
    %0 = bit.shr %arg0, %cst0 : i64
    %1 = bit.shr %arg0:%arg1, %cst0 : i64
    // return %[[ARG0]], %[[ARG0]]
    return %0, %1 : i64, i64
}

// shr x, >=N         = 0
// shr poison, >=N    = 0
// shr x:y, >=2N      = 0
// shr ?:poison, >=2N = 0
// shr poison:?, >=2N = 0
// CHECK-LABEL: func.func @shr_zero(
func.func @shr_zero(%arg0: i64, %arg1: i64) -> (i64, i64, i64, i64, i64) {
    // CHECK-DAG: %[[CST0:.+]] = bit.constant 0 : i64
    %poison = ub.poison : i64
    %cst64 = index.constant 64
    %cst128 = index.constant 128
    %0 = bit.shr %arg0, %cst64 : i64
    %1 = bit.shr %poison, %cst64 : i64
    %2 = bit.shr %arg0:%arg1, %cst128 : i64
    %3 = bit.shr %arg0:%poison, %cst128 : i64
    %4 = bit.shr %poison:%arg1, %cst128 : i64
    // CHECK: return %[[CST0]], %[[CST0]], %[[CST0]], %[[CST0]], %[[CST0]]
    return %0, %1, %2, %3, %4 : i64, i64, i64, i64, i64
}

// shr poison, <N         = poison
// shr poison:?, <2N      = poison
// shr x:poison, <N       = poison
// shr poison:poison, <2N = poison
// CHECK-LABEL: func.func @shr_poison(
// CHECK-SAME: %[[ARG0:.+]]: i64
func.func @shr_poison(%arg0: i64) -> (i64, i64, i64, i64, i64) {
    // CHECK-DAG: %[[POISON:.+]] = ub.poison : i64
    // CHECK-DAG: %[[CST63:.+]] = index.constant 63
    %poisoni64 = ub.poison : i64
    %cst63 = index.constant 63
    %cst127 = index.constant 127
    %0 = bit.shr %poisoni64, %cst63 : i64
    %1 = bit.shr %arg0:%poisoni64, %cst63 : i64
    // CHECK-DAG: %[[SHR:.+]] = bit.shr %[[ARG0]], %[[CST63]]
    %2 = bit.shr %arg0:%poisoni64, %cst127 : i64
    %3 = bit.shr %poisoni64:%arg0, %cst127 : i64
    %4 = bit.shr %poisoni64:%poisoni64, %cst127 : i64
    // CHECK: return %[[POISON]], %[[POISON]], %[[SHR]], %[[POISON]], %[[POISON]]
    return %0, %1, %2, %3, %4 : i64, i64, i64, i64, i64
}

// shr ?, poison   = poison
// shr ?:?, poison = poison
// CHECK-LABEL: func.func @shr_poison_idx(
func.func @shr_poison_idx(%arg0: i64, %arg1: i64) -> (i64, i64, i64, i64, i64) {
    // CHECK-DAG: %[[POISON:.+]] = ub.poison : i64
    %poisonidx = ub.poison : index
    %poisoni64 = ub.poison : i64
    %0 = bit.shr %poisoni64, %poisonidx : i64
    %1 = bit.shr %arg0:%arg1, %poisonidx : i64
    %2 = bit.shr %arg0:%poisoni64, %poisonidx : i64
    %3 = bit.shr %poisoni64:%arg0, %poisonidx : i64
    %4 = bit.shr %poisoni64:%poisoni64, %poisonidx : i64
    // CHECK: return %[[POISON]], %[[POISON]], %[[POISON]], %[[POISON]], %[[POISON]]
    return %0, %1, %2, %3, %4 : i64, i64, i64, i64, i64
}

//===----------------------------------------------------------------------===//
// count
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @count_poison(
func.func @count_poison() -> index {
    // CHECK-DAG: %[[POISON:.+]] = ub.poison : index
    %poison = ub.poison : i64
    %0 = bit.count %poison : i64
    // CHECK: return %[[POISON]]
    return %0 : index
}

//===----------------------------------------------------------------------===//
// clz
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @clz_poison(
func.func @clz_poison() -> index {
    // CHECK-DAG: %[[POISON:.+]] = ub.poison : index
    %poison = ub.poison : i64
    %0 = bit.clz %poison : i64
    // CHECK: return %[[POISON]]
    return %0 : index
}

//===----------------------------------------------------------------------===//
// ctz
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @ctz_poison(
func.func @ctz_poison() -> index {
    // CHECK-DAG: %[[POISON:.+]] = ub.poison : index
    %poison = ub.poison : i64
    %0 = bit.ctz %poison : i64
    // CHECK: return %[[POISON]]
    return %0 : index
}
