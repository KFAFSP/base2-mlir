//===- InterpretableType.td --------------------------------*- tablegen -*-===//
//
// This is the definitions file for the Base2 InterpretableType interface.
//
//===----------------------------------------------------------------------===//

#ifndef BASE2_INTERPRETABLETYPE
#define BASE2_INTERPRETABLETYPE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// InterpretableType
//===----------------------------------------------------------------------===//

class Base2_InterpretMethod<string name>
        : InterfaceMethod<
            /*desc=*/"Implements " # desc # " on constant bit sequences.",
            /*retTy=*/"bit_result",
            /*methodName=*/name,
            /*args=*/(ins
                "const BitSequence &":$lhs,
                "const BitSequence &":$rhs,
                "RoundingMode":$roundingMode),
            /*methodBody=*/"",
            /*defaultImplementation=*/"return std::nullopt;">;

def Base2_InterpretableType : TypeInterface<"InterpretableType"> {
    let cppNamespace = "::mlir::base2";

    let description = [{
        Specialization of a BitSequenceType that has a fixed interpretation of
        its bits.
    }];

    let methods = [
        InterfaceMethod<
            /*desc=*/"Determines whether value_cast is permissible.",
            /*retTy=*/"bool",
            /*methodName=*/"canValueCast",
            /*args=*/(ins "InterpretableType":$from, "InterpretableType":$to),
            /*methodBody=*/"",
            /*defaultImplementation=*/[{
                return interpretable_type_interface_defaults::canValueCast(
                    $_type,
                    from,
                    to);
            }]>,
        InterfaceMethod<
            /*desc=*/"Implements value_cast on constant bit sequences.",
            /*retTy=*/"bit_result",
            /*methodName=*/"valueCast",
            /*args=*/(ins
                "InterpretableType":$from,
                "const BitSequence &":$value,
                "InterpretableType":$to,
                "RoundingMode":$roundingMode),
            /*methodBody=*/"",
            /*defaultImplementation=*/"return std::nullopt;">,
        InterfaceMethod<
            /*desc=*/"Implements cmp on constant bit sequences.",
            /*retTy=*/"cmp_result",
            /*methodName=*/"cmp",
            /*args=*/(ins
                "const BitSequence &":$lhs,
                "const BitSequence &":$rhs),
            /*methodBody=*/"",
            /*defaultImplementation=*/[{
                return interpretable_type_interface_defaults::cmp(
                    $_type,
                    lhs,
                    rhs);
            }]>,
        InterfaceMethod<
            /*desc=*/"Implements min on constant bit sequences.",
            /*retTy=*/"bit_result",
            /*methodName=*/"min",
            /*args=*/(ins
                "const BitSequence &":$lhs,
                "const BitSequence &":$rhs),
            /*methodBody=*/"",
            /*defaultImplementation=*/[{
                return interpretable_type_interface_defaults::min(
                    $_type,
                    lhs,
                    rhs);
            }]>,
        InterfaceMethod<
            /*desc=*/"Implements max on constant bit sequences.",
            /*retTy=*/"bit_result",
            /*methodName=*/"max",
            /*args=*/(ins
                "const BitSequence &":$lhs,
                "const BitSequence &":$rhs),
            /*methodBody=*/"",
            /*defaultImplementation=*/[{
                return interpretable_type_interface_defaults::max(
                    $_type,
                    lhs,
                    rhs);
            }]>,
        Base2_InterpretMethod<"add">,
        Base2_InterpretMethod<"sub">,
        Base2_InterpretMethod<"mul">,
        Base2_InterpretMethod<"div">,
        InterfaceMethod<
            /*desc=*/"Implements mod on constant bit sequences.",
            /*retTy=*/"bit_result",
            /*methodName=*/"mod",
            /*args=*/(ins
                "const BitSequence &":$lhs,
                "const BitSequence &":$rhs),
            /*methodBody=*/"",
            /*defaultImplementation=*/"return std::nullopt;">,
        InterfaceMethod<
            /*desc=*/"Gets ValueFacts for a constant bit sequence.",
            /*retTy=*/"ValueFacts",
            /*methodName=*/"getFacts",
            /*args=*/(ins "const BitSequence &":$value),
            /*methodBody=*/"",
            /*defaultImplementation=*/"return ValueFacts::None;">];

    code extraClassDeclaration = [{
    public:
        /// Gets the bit sequence length.
        [[nodiscard]] bit_width_t getBitWidth() const
        {
            return cast<BitSequenceType>().getBitWidth();
        }
    }];
}

//===----------------------------------------------------------------------===//
// InterpretableLike
//===----------------------------------------------------------------------===//

def Base2_InterpretableLike
        : TypeConstraint<
            /*predicate=*/CPred<"$_self.isa<::mlir::base2::InterpretableLikeType>()">,
            /*summary=*/"Type or container of InterpretableType",
            /*cppClassName=*/"::mlir::base2::InterpretableLikeType">;

#endif
