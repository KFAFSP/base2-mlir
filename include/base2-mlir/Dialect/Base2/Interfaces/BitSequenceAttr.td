//===- BitSequenceAttr.td ----------------------------------*- tablegen -*-===//
//
// This is the definitions file for the Base2 BitSequenceAttr interface.
//
//===----------------------------------------------------------------------===//

#ifndef BASE2_BITSEQUENCEATTR
#define BASE2_BITSEQUENCEATTR

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// BitSequenceAttr
//===----------------------------------------------------------------------===//

def Base2_BitSequenceAttr : AttrInterface<"BitSequenceAttr"> {
    let cppNamespace = "::mlir::base2";

    let description = [{
        Interface for an attribute that declares a compile-time constant
        value of a `BitSequenceType`.
    }];

    let methods = [
        InterfaceMethod<
            /*desc=*/"Gets the BitSequenceType.",
            /*retTy=*/"BitSequenceType",
            /*methodName=*/"getType",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/"">,
        InterfaceMethod<
            /*desc=*/"Gets the stored BitSequence value.",
            /*retTy=*/"BitSequence",
            /*methodName=*/"getValue",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/"">];

    code extraClassDeclaration = [{
    public:
        /// The type of the stored value.
        using ValueType = BitSequence;

        /// Builds the canonical BitSequenceAttr for @p type using @p bits .
        ///
        /// When constructing a BitSequenceAttr, built-in attributes are
        /// prefered where possible. In all other scenarios, a BitsAttr is
        /// returned.
        ///
        /// @pre    `type`
        /// @pre    `bits.size() >= type.getBitWidth()`
        [[nodiscard]] static BitSequenceAttr get(
            BitSequenceType type,
            const BitSequence &bits);
        /// Builds the canonical BitSequenceAttr for @p attr .
        ///
        /// @pre    `attr`
        [[nodiscard]] static BitSequenceAttr get(BitSequenceAttr attr)
        {
            assert(attr);

            return get(attr.getType(), attr.getValue());
        }

        /// Performs a bit cast to @p type .
        ///
        /// @pre    `type`
        /// @pre    `type.getBitWidth() == getType().getBitWidth()`
        [[nodiscard]] BitSequenceAttr bitCast(BitSequenceType type) const
        {
            assert(type);
            assert(type.getBitWidth() == getType().getBitWidth());

            return get(type, getValue());
        }

        /// Applies @p fn to the contained value.
        ///
        /// If @p elementTy is nullptr, getType() is used.
        /// If @p fn returns std::nullopt, the result will be nullptr.
        ///
        /// @pre    bit width of @p elementTy and @p fn result matches
        [[nodiscard]] BitSequenceAttr map(
            UnaryBitSequenceFn fn,
            BitSequenceType elementTy = {}) const
        {
            if (!elementTy) elementTy = getType();

            if (const auto mapped = fn(getValue()))
                return BitSequenceAttr::get(elementTy, *mapped);

            return BitSequenceAttr{};
        }
        /// Combines the values contained in this and @p rhs using @p fn .
        ///
        /// If @p elementTy is nullptr, getType() is used.
        /// If @p fn returns std::nullopt, the result will be nullptr.
        ///
        /// @pre    `rhs`
        /// @pre    bit width of @p elementTy and @p fn result matches
        [[nodiscard]] BitSequenceAttr zip(
            BinaryBitSequenceFn fn,
            BitSequenceAttr rhs,
            BitSequenceType elementTy = {}) const
        {
            assert(rhs);

            if (!elementTy) elementTy = getType();

            if (const auto zipped = fn(getValue(), rhs.getValue()))
                return BitSequenceAttr::get(elementTy, *zipped);

            return BitSequenceAttr{};
        }
    }];
}

//===----------------------------------------------------------------------===//
// DenseBitSequencesAttr
//===----------------------------------------------------------------------===//

def Base2_DenseBitSequencesAttr : Attr<
        /*condition=*/CPred<"$_self.isa<::mlir::base2::DenseBitSequencesAttr>()">,
        /*desc=*/"Constant densely-packed bit sequence container attribute"> {
    code storageType = "::mlir::base2::DenseBitSequencesAttr";
    code returnType = "::mlir::base2::DenseBitSequencesAttr::range";
    code convertFromStorage = "$_self.getValues()";
}

//===----------------------------------------------------------------------===//
// BitSequenceLikeAttr
//===----------------------------------------------------------------------===//

def Base2_BitSequenceLikeAttr : Attr<
        /*condition=*/CPred<"$_self.isa<::mlir::base2::BitSequenceLikeAttr>()">,
        /*desc=*/"Value or container of values of BitSequenceType attribute"> {
    code storageType = "::mlir::base2::BitSequenceLikeAttr";
    code returnType = "::mlir::base2::BitSequenceLikeAttr";
    code convertFromStorage = "$_self";
}

#endif
