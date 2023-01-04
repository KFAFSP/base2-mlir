/// Implements the Base2 dialect enums.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "base2-mlir/Dialect/Base2/Enums.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;
using namespace mlir::base2;

//===- Generated implementation -------------------------------------------===//

#include "base2-mlir/Dialect/Base2/Enums.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PartialOrderingPredicate
//===----------------------------------------------------------------------===//

arith::CmpFPredicate base2::getCmpFPredicate(PartialOrderingPredicate pred)
{
    using underlying_type = std::underlying_type_t<PartialOrderingPredicate>;
    const auto predBits = static_cast<underlying_type>(pred);

    // NOTE: The enumeration is exactly the same, as it is inherited from LLVM.
    return static_cast<arith::CmpFPredicate>(predBits);
}

std::optional<arith::CmpIPredicate>
base2::getCmpIPredicate(Signedness signedness, PartialOrderingPredicate pred)
{
    // First, convert to the strong predicate.
    pred = strong(pred);

    // Then, switch based on the predicate. Only fail for signless where the
    // semantics actually matter.
    switch (pred) {
    case PartialOrderingPredicate::OrderedAndEqual:
        return arith::CmpIPredicate::eq;
    case PartialOrderingPredicate::OrderedAndUnequal:
        return arith::CmpIPredicate::ne;
    case PartialOrderingPredicate::OrderedAndLess:
        switch (signedness) {
        case Signedness::Signless: return std::nullopt;
        case Signedness::Signed: return arith::CmpIPredicate::slt;
        case Signedness::Unsigned: return arith::CmpIPredicate::ult;
        }
    case PartialOrderingPredicate::OrderedAndLessOrEqual:
        switch (signedness) {
        case Signedness::Signless: return std::nullopt;
        case Signedness::Signed: return arith::CmpIPredicate::sle;
        case Signedness::Unsigned: return arith::CmpIPredicate::ule;
        }
    case PartialOrderingPredicate::OrderedAndGreater:
        switch (signedness) {
        case Signedness::Signless: return std::nullopt;
        case Signedness::Signed: return arith::CmpIPredicate::sgt;
        case Signedness::Unsigned: return arith::CmpIPredicate::ugt;
        }
    case PartialOrderingPredicate::OrderedAndGreaterOrEqual:
        switch (signedness) {
        case Signedness::Signless: return std::nullopt;
        case Signedness::Signed: return arith::CmpIPredicate::sge;
        case Signedness::Unsigned: return arith::CmpIPredicate::uge;
        }
    default: return std::nullopt;
    }
}
