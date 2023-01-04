// RUN: base2-opt %s -split-input-file -verify-diagnostics

// expected-error@+1 {{BitSequenceType}}
#bits_not_a_bit_sequence_type = #base2.bits<"0xDEADBEEF"> : memref<3xi32>

// -----

// expected-error@+1 {{bit width}}
#bits_too_few_bits = #base2.bits<"0b0100"> : i5

// -----

// expected-error@+1 {{BitSequenceType}}
#dense_bits_not_a_bit_sequence_type = #base2.dense_bits<tensor<3xvector<3xi32>> = dense<3>>

// -----

// expected-error@+1 {{shape}}
#dense_bits_shape_mismatch = #base2.dense_bits<tensor<3xi32> = dense<[2, 2]>>
