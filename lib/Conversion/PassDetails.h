/// Declaration of the SoftFLoat to libsoftfloat loowering pass.
///
/// @file
/// @author     Jihaong Bi (jiahong.bi@mailbox.tu-dresden.de)

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

// Forward declaration from Dialect.h
template<typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

class AffineDialect;

namespace arith {
class ArithDialect;
} // namespace arith

namespace base2 {
class Base2Dialect;
} // namespace base2

namespace func {
class FuncDialect;
} // namespace func

namespace linalg {
class LinalgDialect;
} // namespace linalg

namespace memref {
class MemRefDialect;
} // namespace memref

namespace scf {
class SCFDialect;
} // namespace scf

namespace softfloat {
class SoftFloatDialect;
} // namespace softfloat

namespace tensor {
class TensorDialect;
} // namespace tensor

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_CLASSES
#include "base2-mlir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir