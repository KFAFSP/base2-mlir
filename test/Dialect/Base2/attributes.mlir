// RUN: base2-opt %s | FileCheck %s

#bits_hex = #base2.bits<"0x0123456789abcdefABCDEF"> : i88
#bits_hex_delim = #base2.bits<"0x_0123456789_abcdef_ABCDEF"> : i88
#bits_hex_slash = #base2.bits<"0x_0123456789_abcdef_ABCDEF/48"> : i48

// CHECK-LABEL: func.func @bits_hex(
// CHECK-SAME: a0 = #base2.bits<"0x0123456789ABCDEFABCDEF"> : i88
// CHECK-SAME: a1 = #base2.bits<"0x0123456789ABCDEFABCDEF"> : i88
// CHECK-SAME: a2 = #base2.bits<"0xABCDEFABCDEF"> : i48
func.func @bits_hex() attributes { a0 = #bits_hex, a1 = #bits_hex_delim, a2 = #bits_hex_slash } {
    return
}

#bits_bin = #base2.bits<"0b00110101"> : i8
#bits_bin_delim = #base2.bits<"0b_0011_0101"> : i8
#bits_bin_slash = #base2.bits<"0b00110101/4"> : i4

// CHECK-LABEL: func.func @bits_bin(
// CHECK-SAME: a0 = #base2.bits<"0x35"> : i8
// CHECK-SAME: a1 = #base2.bits<"0x35"> : i8
// CHECK-SAME: a2 = #base2.bits<"0x05/4"> : i4
func.func @bits_bin() attributes { a0 = #bits_bin, a1 = #bits_bin_delim, a2 = #bits_bin_slash } {
    return
}
