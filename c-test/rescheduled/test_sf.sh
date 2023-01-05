mkdir -p temp && cd temp
~/Desktop/base2dialect/build/bin/base2-opt ../rescheduled_softfloat.mlir -convert-softfloat-to-lib -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-func-to-llvm -convert-memref-to-llvm -reconcile-unrealized-casts --llvm-legalize-for-export -o test_sf.mlir
~/Desktop/llvm/bin/mlir-translate test_sf.mlir -mlir-to-llvmir -o test_sf.ll
~/Desktop/llvm/bin/llc test_sf.ll -filetype=obj -o test_sf.o