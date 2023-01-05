mkdir -p temp && cd temp
~/Desktop/base2dialect/build/bin/base2-opt ../rescheduled.mlir -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-func-to-llvm -convert-memref-to-llvm -reconcile-unrealized-casts --llvm-legalize-for-export -o test_std.mlir
~/Desktop/llvm/bin/mlir-translate test_std.mlir -mlir-to-llvmir -o test_std.ll
~/Desktop/llvm/bin/llc test_std.ll -filetype=obj -o test_std.o