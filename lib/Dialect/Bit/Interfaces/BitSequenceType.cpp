/// Implements the Bit dialect BitSequenceType interface.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "base2-mlir/Dialect/Bit/Interfaces/BitSequenceType.h"

#include "base2-mlir/Dialect/Bit/IR/Bit.h"

using namespace mlir;
using namespace mlir::bit;

//===- Generated implementation -------------------------------------------===//

#include "base2-mlir/Dialect/Bit/Interfaces/BitSequenceType.cpp.inc"

//===----------------------------------------------------------------------===//

namespace {

template<class Float>
struct FloatModel : BitSequenceType::ExternalModel<FloatModel<Float>, Float> {
    static bit_width_t getBitWidth(Type type)
    {
        return type.cast<FloatType>().getWidth();
    }
};

struct IntModel : BitSequenceType::ExternalModel<IntModel, IntegerType> {
    static bit_width_t getBitWidth(Type type)
    {
        return type.cast<IntegerType>().getWidth();
    }
};

} // namespace

void bit::registerBitSequenceTypeModels(MLIRContext &ctx)
{
    Float8E5M2Type::attachInterface<FloatModel<Float8E5M2Type>>(ctx);
    BFloat16Type::attachInterface<FloatModel<BFloat16Type>>(ctx);
    Float16Type::attachInterface<FloatModel<Float16Type>>(ctx);
    Float32Type::attachInterface<FloatModel<Float32Type>>(ctx);
    Float64Type::attachInterface<FloatModel<Float64Type>>(ctx);
    Float80Type::attachInterface<FloatModel<Float80Type>>(ctx);
    Float128Type::attachInterface<FloatModel<Float128Type>>(ctx);

    IntegerType::attachInterface<IntModel>(ctx);
}
