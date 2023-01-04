/// Main entry point for the base2-mlir optimizer driver.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "base2-mlir/Dialect/Base2/IR/Base2.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

int main(int argc, char* argv[])
{
    DialectRegistry registry;
    registerAllDialects(registry);

    registry.insert<base2::Base2Dialect>();

    registerAllPasses();

    return asMainReturnCode(
        MlirOptMain(argc, argv, "base2-mlir optimizer driver\n", registry));
}
