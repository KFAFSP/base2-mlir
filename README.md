# `base2-mlir`

This repository implements a set of MLIR dialects dealing with different representations of binary numbers.

## Building

The `base2-mlir` project is built using **CMake** (version `3.20` or newer). Make sure to provide all dependencies required by the project, either by installing them to system-default locations, or by setting the appropriate search location hints!

```sh
# Configure.
cmake -S . -B build \
    -G Ninja \
    -DLLVM_DIR=$LLVM_PREFIX/lib/cmake/llvm \
    -DMLIR_DIR=$MLIR_PREFIX/lib/cmake/mlir

# Build.
cmake --build build
```

The following CMake variables can be configured:

|       Name | Type     | Description |
| ---------: | :------- | --- |
| `LLVM_DIR` | `STRING` | Path to the CMake directory of an **LLVM** installation. <br/> *e.g. `~/tools/llvm-17/lib/cmake/llvm`* |
| `MLIR_DIR` | `STRING` | Path to the CMake directory of an **MLIR** installation. <br/> *e.g. `~/tools/llvm-17/lib/cmake/mlir`* |

## License

This project is licensed under the ISC license.
