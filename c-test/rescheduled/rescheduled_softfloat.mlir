#le_10 = affine_set<(i): (10 - i >= 0)>
#ge_10 = affine_set<(i): (i - 10 >= 0)>
#eq_0_le_10 = affine_set<(i, j): (i == 0, 10 - j >= 0)>
#eq_10 = affine_set<(i): (i - 10 == 0)>
!scalar = !softfloat.sfloat

module {

    func.func @cast_float(%a: f64) -> !scalar attributes {llvm.emit_c_interface} {
        %exp_bits = arith.constant 11 : i8
        %frac_bits = arith.constant 52 : i8
        %exp_bias = arith.constant -1023 : i32
        %true = arith.constant true
        %sign = arith.constant -1 : i8

        %1 = softfloat.castfloat %a ( %exp_bits, %frac_bits, %exp_bias, %true, %true, %true, %true, %sign ) : !scalar

        return %1 : !scalar
    }

    func.func @kernel_sf(%S: memref<11x11x!scalar>{ llvm.name = "S" }, %D: memref<11x11x11x!scalar>{ llvm.name = "D" }, %u: memref<11x11x11x!scalar>{ llvm.name = "u" }, %v: memref<11x11x11x!scalar>{ llvm.name = "v" }, %t: memref<11x11x11x!scalar>{ llvm.name = "t" }, %r: memref<11x11x11x!scalar>{ llvm.name = "r" }, %t0:  memref<11x11x11x!scalar>{ llvm.name = "t0" }, %t1: memref<11x11x11x!scalar>{ llvm.name = "t1" }, %t2: memref<11x11x11x!scalar>{ llvm.name = "t2" }, %t3: memref<11x11x11x!scalar>{ llvm.name = "t3" }) attributes {llvm.emit_c_interface} {
        %fzero = arith.constant 0.0 : f64
        %zero = func.call @cast_float(%fzero) : (f64) -> !scalar
        %exp_bits = arith.constant 11 : i8
        %frac_bits = arith.constant 52 : i8
        %exp_bias = arith.constant -1023 : i32
        %true = arith.constant true
        %sign = arith.constant -1 : i8

        affine.for %c1 = 0 to 11 {
            affine.for %c2 = 0 to 11 {
                affine.for %c3 = 0 to 11 {
                    affine.store %zero, %t1[%c1,%c2,%c3] : memref<11x11x11x!scalar>
                    affine.for %c4 = 0 to 21 {
                        affine.if #le_10(%c4) {
                            %1 = affine.load %S[%c1,%c4] : memref<11x11x!scalar>
                            %2 = affine.load %u[%c2,%c3,%c4] : memref<11x11x11x!scalar>
                            %3 = softfloat.mul %1, %2 ( %exp_bits, %frac_bits, %exp_bias, %true, %true, %true, %true, %sign ) : !scalar
                            %4 = affine.load %t1[%c1,%c2,%c3] : memref<11x11x11x!scalar>
                            %5 = softfloat.add %3, %4 ( %exp_bits, %frac_bits, %exp_bias, %true, %true, %true, %true, %sign ) : !scalar
                            affine.store %5, %t1[%c1,%c2,%c3] : memref<11x11x11x!scalar>
                        }
                        affine.if #ge_10(%c4) {
                            %1 = affine.load %S[%c3,%c4 - 10] : memref<11x11x!scalar>
                            %2 = affine.load %t1[%c1,%c2,%c3] : memref<11x11x11x!scalar>
                            %3 = softfloat.mul %1, %2 ( %exp_bits, %frac_bits, %exp_bias, %true, %true, %true, %true, %sign ) : !scalar
                            %4 = affine.load %t0[%c4 - 10,%c1,%c2] : memref<11x11x11x!scalar>
                            %5 = softfloat.add %3, %4 ( %exp_bits, %frac_bits, %exp_bias, %true, %true, %true, %true, %sign ) : !scalar
                            affine.store %5, %t0[%c4 - 10,%c1,%c2] : memref<11x11x11x!scalar>
                        }
                        affine.if #eq_0_le_10(%c3,%c4) {
                            affine.store %zero, %t0[%c4,%c1,%c2] : memref<11x11x11x!scalar>
                        }
                    }
                }
            }
        }

        affine.for %c1 = 0 to 11 {
            affine.for %c2 = 0 to 11 {
                affine.for %c3 = 0 to 11 {
                    affine.store %zero, %t[%c1,%c2,%c3] : memref <11x11x11x!scalar>
                    affine.for %c4 = 0 to 21 {
                        affine.if #le_10(%c4) {
                            %1 = affine.load %S[%c1,%c4] : memref<11x11x!scalar>
                            %2 = affine.load %t0[%c2,%c3,%c4] : memref<11x11x11x!scalar>
                            %3 = softfloat.mul %1, %2 ( %exp_bits, %frac_bits, %exp_bias, %true, %true, %true, %true, %sign ) : !scalar
                            %4 = affine.load %t[%c1,%c2,%c3] : memref<11x11x11x!scalar>
                            %5 = softfloat.add %3, %4 ( %exp_bits, %frac_bits, %exp_bias, %true, %true, %true, %true, %sign ) : !scalar
                            affine.store %5, %t[%c1,%c2,%c3] : memref<11x11x11x!scalar>
                            affine.if #eq_10(%c4) {
                                %6 = affine.load %D[%c1,%c2,%c3]: memref<11x11x11x!scalar>
                                %7 = affine.load %t[%c1,%c2,%c3]: memref<11x11x11x!scalar>
                                %8 = softfloat.mul %6, %7 ( %exp_bits, %frac_bits, %exp_bias, %true, %true, %true, %true, %sign ) : !scalar
                                affine.store %8, %r[%c1,%c2,%c3] : memref<11x11x11x!scalar>
                            }
                        }
                        affine.if #ge_10(%c4) {
                            %1 = affine.load %S[%c3,%c4 - 10] : memref<11x11x!scalar>
                            %2 = affine.load %t[%c1,%c2,%c3] : memref<11x11x11x!scalar>
                            %3 = softfloat.mul %1, %2 ( %exp_bits, %frac_bits, %exp_bias, %true, %true, %true, %true, %sign ) : !scalar
                            %4 = affine.load %t3[%c4 - 10,%c1,%c2] : memref<11x11x11x!scalar>
                            %5 = softfloat.add %3, %4 ( %exp_bits, %frac_bits, %exp_bias, %true, %true, %true, %true, %sign ) : !scalar
                            affine.store %5, %t3[%c4 - 10,%c1,%c2] : memref<11x11x11x!scalar>
                        }
                        affine.if #eq_0_le_10(%c3,%c4) {
                            affine.store %zero, %t3[%c4,%c1,%c2] : memref<11x11x11x!scalar>
                        }
                    }
                }
            }
        }

        affine.for %c1 = 0 to 11 {
            affine.for %c2 = 0 to 11 {
                affine.for %c3 = 0 to 11 {
                    affine.for %c4 = 0 to 11 {
                        affine.if #eq_10(%c3) {
                            affine.store %zero, %v[%c4,%c1,%c2] : memref<11x11x11x!scalar>
                        }
                        affine.if #eq_10(%c4) {
                            affine.store %zero, %t2[%c1,%c2,%c3] : memref<11x11x11x!scalar>
                        }
                        %1 = affine.load %S[%c4,%c1] : memref<11x11x!scalar>
                        %2 = affine.load %t3[%c2,%c3,%c4] : memref<11x11x11x!scalar>
                        %3 = softfloat.mul %1, %2 ( %exp_bits, %frac_bits, %exp_bias, %true, %true, %true, %true, %sign ) : !scalar
                        %4 = affine.load %t2[%c1,%c2,%c3] : memref<11x11x11x!scalar>
                        %5 = softfloat.add %3, %4 ( %exp_bits, %frac_bits, %exp_bias, %true, %true, %true, %true, %sign ) : !scalar
                        affine.store %5, %t2[%c1,%c2,%c3] : memref<11x11x11x!scalar>
                    }
                    affine.for %c4 = 10 to 21 {
                        %1 = affine.load %S[%c3,%c4 - 10] : memref<11x11x!scalar>
                        %2 = affine.load %t2[%c1,%c2,%c3] : memref<11x11x11x!scalar>
                        %3 = softfloat.mul %1, %2 ( %exp_bits, %frac_bits, %exp_bias, %true, %true, %true, %true, %sign ) : !scalar
                        %4 = affine.load %v[%c4 - 10,%c1,%c2] : memref<11x11x11x!scalar>
                        %5 = softfloat.add %3, %4 ( %exp_bits, %frac_bits, %exp_bias, %true, %true, %true, %true, %sign ) : !scalar
                        affine.store %5, %v[%c4 - 10,%c1,%c2] : memref<11x11x11x!scalar>
                    }
                }
            }
        }

        return
    }
}