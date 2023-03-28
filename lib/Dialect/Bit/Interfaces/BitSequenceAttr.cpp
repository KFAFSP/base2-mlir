/// Implements the Bit dialect BitSequenceAttr interface.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "base2-mlir/Dialect/Bit/Interfaces/BitSequenceAttr.h"

#include "base2-mlir/Dialect/Bit/IR/Bit.h"

#include "llvm/ADT/TypeSwitch.h"

#include <algorithm>
#include <numeric>

using namespace mlir;
using namespace mlir::bit;

//===- Generated implementation -------------------------------------------===//

#include "base2-mlir/Dialect/Bit/Interfaces/BitSequenceAttr.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// BitSequenceAttr
//===----------------------------------------------------------------------===//

BitSequenceAttr
BitSequenceAttr::get(BitSequenceType type, const BitSequence &bits)
{
    assert(type);
    assert(type.getBitWidth() <= bits.size());

    const auto bitWidth = type.getBitWidth();
    const auto raw = bits.asUInt().trunc(bitWidth);

    return TypeSwitch<Type, Attribute>(type)
        .template Case<IntegerType>([&](IntegerType intTy) -> BitSequenceAttr {
            // NOTE: MLIR APInt literals cannot parse more than 96 bits in
            //       hexadecimal. Types exceeding that length _must_ be encoded
            //       as a BitsAttr!
            if (bits.size() > 96) return BitsAttr::get(type, bits);
            return IntegerAttr::get(intTy, raw);
        })
        .template Case<FloatType>([&](FloatType fltTy) {
            return FloatAttr::get(
                fltTy,
                APFloat(fltTy.getFloatSemantics(), raw));
        })
        .Default(
            [&](BitSequenceType type) { return BitsAttr::get(type, raw); });
}

//===----------------------------------------------------------------------===//
// DenseBitSequencesAttr
//===----------------------------------------------------------------------===//

bool DenseBitSequencesAttr::classof(ElementsAttr attr)
{
    // Must have a BitSequenceType element type.
    const auto elementTy = attr.getElementType().dyn_cast<BitSequenceType>();
    if (!elementTy.isa<BitSequenceType>()) return false;

    // BitSequence iteration always takes precedence.
    if (attr.try_value_begin<BitSequence>()) return true;

    // For a FloatType, it must be APFloat iterable.
    if (elementTy.isa<FloatType>())
        return attr.try_value_begin<llvm::APFloat>().has_value();

    // Otherwise, it must be APInt iterable.
    return attr.try_value_begin<llvm::APInt>().has_value();
}

DenseIntOrFPElementsAttr
DenseBitSequencesAttr::getData(ShapedType type, ArrayRef<BitSequence> values)
{
    assert(type);
    assert(type.hasStaticShape());
    assert(
        values.size() == 1
        || values.size() == static_cast<std::size_t>(type.getNumElements()));
    const auto elementTy = type.getElementType().dyn_cast<BitSequenceType>();
    assert(elementTy);

    // Infer the data attribute type.
    const auto dataElementTy =
        elementTy.isa<IntegerType, FloatType>()
            ? elementTy
            : IntegerType::get(type.getContext(), elementTy.getBitWidth());
    const auto dataTy = type.cloneWith(std::nullopt, dataElementTy);

    // Serialize the values into a raw data buffer in MLIR layout.
    SmallVector<std::uint8_t> rawData;
    if (dataElementTy.isa<FloatType>()) {
        // Float elements must always be serialized as big-endian.
        getBytes(rawData, values, std::endian::big);
    } else {
        // Integer elements are serialized as platform-specific integers, i.e.,
        // in the native endianness.
        if (values.size() == 1 && values.front().size() == 1) {
            // This special case ensures that boolean splats are correctly
            // serialized.
            rawData.push_back(values.front().isZeros() ? 0x00 : 0xFF);
        } else
            getBytes(rawData, values);
    }

    // Build the data attribute.
    const auto view = ArrayRef<char>(
        reinterpret_cast<const char*>(rawData.data()),
        rawData.size());
    return DenseElementsAttr::getFromRawBuffer(dataTy, view)
        .cast<DenseIntOrFPElementsAttr>();
}

DenseBitSequencesAttr
DenseBitSequencesAttr::get(ShapedType type, DenseIntOrFPElementsAttr data)
{
    assert(type);
    assert(data);
    assert(type.getShape().equals(data.getType().getShape()));
    const auto elementTy = type.getElementType().dyn_cast<BitSequenceType>();
    assert(elementTy);
    const auto bitWidth = elementTy.getBitWidth();
    assert(data.getType().getElementTypeBitWidth() == bitWidth);

    // If possible, attempt a bit cast.
    if (elementTy.isa<IntegerType, FloatType>())
        return data.bitcast(elementTy).cast<DenseBitSequencesAttr>();

    // Otherwise, wrap signless integer values in a DenseBitsAttr.
    return DenseBitsAttr::get(
        type,
        data.bitcast(IntegerType::get(type.getContext(), bitWidth))
            .cast<DenseIntElementsAttr>());
}

DenseBitSequencesAttr
DenseBitSequencesAttr::bitCastElements(BitSequenceType elementTy) const
{
    assert(elementTy);
    assert(elementTy.getBitWidth() == getElementType().getBitWidth());

    // Infer the type of the result.
    const auto outTy = getType().cloneWith(std::nullopt, elementTy);

    // Try to get the underlying data attribute.
    auto dataAttr = dyn_cast<DenseIntOrFPElementsAttr>();
    if (!dataAttr) {
        if (const auto denseBits = dyn_cast<DenseBitsAttr>())
            dataAttr = denseBits.getData();
    }

    // Perform bitcasting on the data attribute if we have it.
    if (dataAttr) return get(outTy, dataAttr);

    // If no data attribute was found, we need to take the long route.
    return get(outTy, llvm::to_vector(getValues()));
}

DenseBitSequencesAttr DenseBitSequencesAttr::map(
    UnaryBitSequenceFn fn,
    BitSequenceType elementTy) const
{
    // Infer the type of the result.
    if (!elementTy) elementTy = getElementType();
    const auto outTy = getType().cloneWith(std::nullopt, elementTy);

    // If this is a splat value, we can produce a new splat.
    if (isSplat())
        return DenseBitSequencesAttr::get(outTy, fn(getSplatValue()));

    // Allocate storage for the result elements.
    SmallVector<BitSequence> result;
    result.reserve(size());

    // Map all the elements.
    const auto end = value_end();
    for (auto it = value_begin(); it != end; ++it) result.push_back(fn(*it));

    // Construct the canonical attribute.
    return DenseBitSequencesAttr::get(outTy, result);
}

DenseBitSequencesAttr DenseBitSequencesAttr::zip(
    BinaryBitSequenceFn fn,
    DenseBitSequencesAttr rhs,
    BitSequenceType elementTy) const
{
    assert(rhs);
    assert(getType().getShape().equals(rhs.getType().getShape()));

    // Infer the type of the result.
    if (!elementTy) elementTy = getElementType();
    const auto outTy = getType().cloneWith(std::nullopt, elementTy);

    // If both are splats, we can create a new splat.
    if (isSplat() && rhs.isSplat()) {
        return DenseBitSequencesAttr::get(
            outTy,
            fn(getSplatValue(), rhs.getSplatValue()));
    }

    // Allocate storage for the result elements.
    SmallVector<BitSequence> result;
    result.reserve(size());

    // Zip together the elements.
    const auto lhsEnd = value_end();
    for (auto [lhsIt, rhsIt] = std::make_pair(value_begin(), rhs.value_begin());
         lhsIt != lhsEnd;
         ++lhsIt, ++rhsIt)
        result.push_back(fn(*lhsIt, *rhsIt));

    // Construct the canonical attribute.
    return DenseBitSequencesAttr::get(outTy, result);
}

DenseBitSequencesAttr DenseBitSequencesAttr::zip(
    TernaryBitSequenceFn fn,
    DenseBitSequencesAttr arg1,
    DenseBitSequencesAttr arg2,
    BitSequenceType elementTy) const
{
    assert(arg1 && arg2);
    assert(getType().getShape().equals(arg1.getType().getShape()));
    assert(getType().getShape().equals(arg2.getType().getShape()));

    // Infer the type of the result.
    if (!elementTy) elementTy = getElementType();
    const auto outTy = getType().cloneWith(std::nullopt, elementTy);

    // If all are splats, we can create a new splat.
    if (isSplat() && arg1.isSplat() && arg2.isSplat()) {
        return DenseBitSequencesAttr::get(
            outTy,
            fn(getSplatValue(), arg1.getSplatValue(), arg2.getSplatValue()));
    }

    // Allocate storage for the result elements.
    SmallVector<BitSequence> result;
    result.reserve(size());

    // Zip together the elements.
    const auto end0 = value_end();
    for (auto [it0, it1, it2] =
             std::tuple(value_begin(), arg1.value_begin(), arg2.value_begin());
         it0 != end0;
         ++it0, ++it1, ++it2)
        result.push_back(fn(*it0, *it1, *it2));

    // Construct the canonical attribute.
    return DenseBitSequencesAttr::get(outTy, result);
}

DenseBitSequencesAttr::iterator DenseBitSequencesAttr::value_begin() const
{
    // BitSequence iteration always takes precedence.
    if (const auto values = ElementsAttr::try_value_begin<BitSequence>())
        return *values;

    const auto elementTy = getElementType();

    // For a FloatType, we must use the APFloat iteration.
    if (elementTy.isa<FloatType>()) {
        auto rawIt = try_value_begin<llvm::APFloat>();
        assert(rawIt.has_value());

        auto indexer = mlir::detail::ElementsAttrIndexer::nonContiguous(
            isSplat(),
            llvm::map_iterator(
                std::move(*rawIt),
                [](llvm::APFloat raw) -> BitSequence {
                    return raw.bitcastToAPInt();
                }));

        return DenseBitSequencesAttr::iterator(std::move(indexer), 0);
    }

    // Otherwise, we must use the APInt iteration.
    auto rawIt = try_value_begin<llvm::APInt>();
    assert(rawIt.has_value());

    const auto bitWidth =
        getElementType().cast<BitSequenceType>().getBitWidth();
    auto indexer = mlir::detail::ElementsAttrIndexer::nonContiguous(
        isSplat(),
        llvm::map_iterator(
            std::move(*rawIt),
            [bitWidth](llvm::APInt raw) -> BitSequence {
                return raw.trunc(bitWidth);
            }));

    return DenseBitSequencesAttr::iterator(std::move(indexer), 0);
}

//===----------------------------------------------------------------------===//
// External models
//===----------------------------------------------------------------------===//

namespace {

struct FloatModel : BitSequenceAttr::ExternalModel<FloatModel, FloatAttr> {
    static BitSequenceType getType(Attribute attr)
    {
        return attr.cast<FloatAttr>().getType().cast<BitSequenceType>();
    }
    static BitSequence getValue(Attribute attr)
    {
        const auto bitWidth =
            attr.cast<FloatAttr>().getType().getIntOrFloatBitWidth();
        return attr.cast<FloatAttr>().getValue().bitcastToAPInt().trunc(
            bitWidth);
    }
};

struct IntModel : BitSequenceAttr::ExternalModel<IntModel, IntegerAttr> {
    static BitSequenceType getType(Attribute attr)
    {
        // NOTE: IntegerAttr also stores IndexType. This break the whole
        //       interface-based opt-in mechanism, because we can't decline at
        //       this point.
        // BUG:  Disallow IndexType without crashing.
        return attr.cast<IntegerAttr>().getType().cast<BitSequenceType>();
    }
    static BitSequence getValue(Attribute attr)
    {
        const auto bitWidth =
            attr.cast<IntegerAttr>().getType().getIntOrFloatBitWidth();
        return attr.cast<IntegerAttr>().getValue().trunc(bitWidth);
    }
};

} // namespace

void bit::registerBitSequenceAttrModels(MLIRContext &ctx)
{
    FloatAttr::attachInterface<FloatModel>(ctx);

    IntegerAttr::attachInterface<IntModel>(ctx);
}
