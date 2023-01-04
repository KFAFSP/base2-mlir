/// Implements the Base2 dialect attributes.
///
/// @file
/// @author     Jihaong Bi (jiahong.bi@mailbox.tu-dresden.de)
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "base2-mlir/Dialect/Base2/IR/Attributes.h"

#include "base2-mlir/Dialect/Base2/IR/Base2.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::base2;

//===- Generated implementation -------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "base2-mlir/Dialect/Base2/IR/Attributes.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// BitsAttr
//===----------------------------------------------------------------------===//

LogicalResult base2::BitsAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    Type type,
    BitSequence value)
{
    // Type must be a BitSequenceType.
    const auto bitSequenceTy = type.dyn_cast<BitSequenceType>();
    if (!bitSequenceTy)
        return emitError() << "`" << type << "` is not a `BitSequenceType`";

    // Bit widths must match.
    if (value.size() != bitSequenceTy.getBitWidth())
        return emitError()
               << "bit width (" << value.size() << ") does not match result ("
               << bitSequenceTy.getBitWidth() << ")";

    return success();
}

//===----------------------------------------------------------------------===//
// DenseBitsAttr
//===----------------------------------------------------------------------===//

Attribute DenseBitsAttr::parse(AsmParser &parser, Type)
{
    // `<`
    if (parser.parseLess()) return Attribute{};

    // $type
    const auto typeLoc = parser.getCurrentLocation();
    Type type;
    if (parser.parseType(type)) return Attribute{};
    // `=`
    if (parser.parseEqual()) return Attribute{};

    // Type must be a ShapedType.
    const auto shapedTy = type.dyn_cast<ShapedType>();
    if (!shapedTy) {
        parser.emitError(typeLoc) << "`" << type << "` is not a `ShapedType`";
        return Attribute{};
    }

    // Shape must be static.
    if (!shapedTy.hasStaticShape()) {
        parser.emitError(typeLoc)
            << "`" << type << "` does not have a static shape";
        return Attribute{};
    }

    // Element type must be a BitSequenceType.
    const auto elementTy =
        shapedTy.getElementType().dyn_cast<BitSequenceType>();
    if (!elementTy) {
        parser.emitError(typeLoc) << "`" << shapedTy.getElementType()
                                  << "` is not a `BitSequenceType`";
        return Attribute{};
    }

    // The DenseIntElementsAttr stores signless integers.
    const auto denseTy = shapedTy.cloneWith(
        std::nullopt,
        IntegerType::get(parser.getContext(), elementTy.getBitWidth()));

    // $data
    DenseIntElementsAttr data;
    if (parser.parseAttribute(data, denseTy)) return Attribute{};

    // `>`
    if (parser.parseGreater()) return Attribute{};

    return parser.getChecked<DenseBitsAttr>(
        parser.getContext(),
        shapedTy,
        data);
}

void DenseBitsAttr::print(AsmPrinter &printer) const
{
    // `<`
    printer << "<";

    // $type
    printer << getType();
    // `=`
    printer << " = ";

    // $data
    printer.printAttributeWithoutType(getData());

    // `>`
    printer << ">";
}

LogicalResult DenseBitsAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    Type type,
    DenseIntElementsAttr data)
{
    // Type must be a ShapedType.
    const auto shapedTy = type.dyn_cast_or_null<ShapedType>();
    if (!shapedTy)
        return emitError() << "`" << type << "` is not a `ShapedType`";

    // Shape must be static.
    if (!shapedTy.hasStaticShape())
        return emitError() << "`" << type << "` does not have a static shape";

    // Element type must be a BitSequenceType.
    const auto elementTy =
        shapedTy.getElementType().dyn_cast<BitSequenceType>();
    if (!elementTy)
        return emitError() << "`" << shapedTy.getElementType()
                           << "` is not a `BitSequenceType`";

    // Data must be present.
    if (!data) return emitError() << "expected data";

    // Shapes must match.
    if (!data.getType().getShape().equals(shapedTy.getShape())) {
        auto diag = emitError() << "data shape {";
        const auto printSz = [&](auto sz) {
            if (ShapedType::isDynamic(sz))
                diag << "?";
            else
                diag << sz;
        };
        llvm::interleaveComma(data.getType().getShape(), diag, printSz);
        diag << "} does not match result shape {";
        llvm::interleaveComma(shapedTy.getShape(), diag);
        return diag << "}";
    }

    // Data must store appropriately sized signless integers.
    const auto bitWidth = elementTy.getBitWidth();
    if (!data.getElementType().isSignlessInteger(bitWidth))
        return emitError() << "expected i" << bitWidth << " data";

    return success();
}

//===----------------------------------------------------------------------===//
// Base2Dialect
//===----------------------------------------------------------------------===//

void Base2Dialect::registerAttributes()
{
    addAttributes<
#define GET_ATTRDEF_LIST
#include "base2-mlir/Dialect/Base2/IR/Attributes.cpp.inc"
        >();
}
