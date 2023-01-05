mkdir -p temp && cd temp
~/Desktop/base2dialect/build/bin/base2-opt ../ops.mlir -convert-softfloat-to-lib -convert-func-to-llvm -convert-arith-to-llvm -reconcile-unrealized-casts --llvm-legalize-for-export -o test.mlir
~/Desktop/llvm/bin/mlir-translate test.mlir -mlir-to-llvmir -o test.ll
~/Desktop/llvm/bin/llc test.ll -filetype=obj -o test.o