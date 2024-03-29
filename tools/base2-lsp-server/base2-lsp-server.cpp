/// Main entry point for the base2-mlir MLIR language server.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "base2-mlir/Dialect/Base2/IR/Base2.h"
#include "base2-mlir/Dialect/Bit/IR/Bit.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"
#include "ub-mlir/Dialect/UB/IR/UB.h"

using namespace mlir;

static int asMainReturnCode(LogicalResult r)
{
    return r.succeeded() ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main(int argc, char* argv[])
{
    DialectRegistry registry;
    registerAllDialects(registry);

    registry.insert<base2::Base2Dialect>();
    registry.insert<bit::BitDialect>();
    registry.insert<ub::UBDialect>();

    return asMainReturnCode(MlirLspServerMain(argc, argv, registry));
}
