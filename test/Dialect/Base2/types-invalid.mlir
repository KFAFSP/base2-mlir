// RUN: base2-opt %s -split-input-file -verify-diagnostics

//===----------------------------------------------------------------------===//
// fixed_point
//===----------------------------------------------------------------------===//

// expected-error@+1 {{total bits}}
func.func private @fixed_total_bits() -> !base2.fixed_point<16777215,5>

// -----

//===----------------------------------------------------------------------===//
// ieee754
//===----------------------------------------------------------------------===//

// expected-error@+1 {{precision}}
func.func private @ieee754_exp_range() -> !base2.ieee754<0,8>

// -----

// expected-error@+1 {{exponent bits}}
func.func private @ieee754_exp_range() -> !base2.ieee754<23,0>

// -----

// expected-error@+1 {{exponent bits}}
func.func private @ieee754_exp_range() -> !base2.ieee754<23,33>

// -----

// expected-error@+1 {{bias}}
func.func private @ieee754_bias_limit() -> !base2.ieee754<23,4,-120>

// -----

// expected-error@+1 {{total bits}}
func.func private @ieee754_exp_range() -> !base2.ieee754<16777215,31>
