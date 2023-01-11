mkdir -p temp && cd temp
~/Desktop/base2-mlir/build/bin/base2-opt ../matmul.mlir --memref-expand --convert-linalg-to-loops --convert-softfloat-to-lib --convert-memref-to-llvm --convert-scf-to-cf --convert-func-to-llvm --canonicalize --llvm-legalize-for-export -o test.mlir
~/Desktop/llvm/bin/mlir-translate test.mlir --mlir-to-llvmir -o test.ll
~/Desktop/llvm/bin/llc test.ll -filetype=obj -o test.o