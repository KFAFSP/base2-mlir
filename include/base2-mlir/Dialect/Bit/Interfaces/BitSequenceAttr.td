//===- BitSequenceAttr.td ----------------------------------*- tablegen -*-===//
//
// This is the definitions file for the Bit BitSequenceAttr interface.
//
//===----------------------------------------------------------------------===//

#ifndef BIT_BITSEQUENCEATTR
#define BIT_BITSEQUENCEATTR

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// BitSequenceAttr
//===----------------------------------------------------------------------===//

def Bit_BitSequenceAttr : AttrInterface<"BitSequenceAttr"> {
    let cppNamespace = "::mlir::bit";

    let description = [{
        Interface for an attribute that declares a compile-time constant
        value of a `BitSequenceType`.
    }];

    let methods = [
        InterfaceMethod<
            /*desc=*/"Gets the BitSequenceType.",
            /*retTy=*/"Type",
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

            return get(attr.getType().cast<BitSequenceType>(), attr.getValue());
        }

        /// Performs a bit cast to @p type .
        ///
        /// @pre    `type`
        /// @pre    `type.getBitWidth() == getType().getBitWidth()`
        [[nodiscard]] BitSequenceAttr bitCast(BitSequenceType type) const
        {
            assert(type);
            assert(type.getBitWidth() == getType().cast<BitSequenceType>().getBitWidth());

            return get(type, getValue());
        }

        /// Applies @p fn to the contained value.
        ///
        /// If @p elementTy is nullptr, getType() is used.
        ///
        /// @pre    bit width of @p elementTy and @p fn result matches
        [[nodiscard]] BitSequenceAttr map(
            UnaryBitSequenceFn fn,
            BitSequenceType elementTy = {}) const
        {
            if (!elementTy) elementTy = getType().cast<BitSequenceType>();

            return BitSequenceAttr::get(elementTy, fn(getValue()));
        }
        /// Combines the values contained in this and @p rhs using @p fn .
        ///
        /// If @p elementTy is nullptr, getType() is used.
        ///
        /// @pre    `rhs`
        /// @pre    bit width of @p elementTy and @p fn result matches
        [[nodiscard]] BitSequenceAttr zip(
            BinaryBitSequenceFn fn,
            BitSequenceAttr rhs,
            BitSequenceType elementTy = {}) const
        {
            assert(rhs);

            if (!elementTy) elementTy = getType().cast<BitSequenceType>();

            return BitSequenceAttr::get(
                elementTy,
                fn(getValue(), rhs.getValue()));
        }
        /// Combines the values of this, @p arg1 and @p arg2 using @p fn .
        ///
        /// If @p elementTy is nullptr, getType() is used.
        ///
        /// @pre    `arg1 && arg2`
        /// @pre    bit width of @p elementTy and @p fn result matches
        [[nodiscard]] BitSequenceAttr zip(
            TernaryBitSequenceFn fn,
            BitSequenceAttr arg1,
            BitSequenceAttr arg2,
            BitSequenceType elementTy = {}) const
        {
            assert(arg1 && arg2);

            if (!elementTy) elementTy = getType().cast<BitSequenceType>();

            return BitSequenceAttr::get(
                elementTy,
                fn(getValue(), arg1.getValue(), arg2.getValue()));
        }
    }];
}

//===----------------------------------------------------------------------===//
// DenseBitSequencesAttr
//===----------------------------------------------------------------------===//

def Bit_DenseBitSequencesAttr : Attr<
        /*condition=*/CPred<"$_self.isa<::mlir::bit::DenseBitSequencesAttr>()">,
        /*desc=*/"Constant densely-packed bit sequence container attribute"> {
    code storageType = "::mlir::bit::DenseBitSequencesAttr";
    code returnType = "::mlir::bit::DenseBitSequencesAttr::range";
    code convertFromStorage = "$_self.getValues()";
}

//===----------------------------------------------------------------------===//
// BitSequenceLikeAttr
//===----------------------------------------------------------------------===//

def Bit_BitSequenceLikeAttr : Attr<
        /*condition=*/CPred<"$_self.isa<::mlir::bit::BitSequenceLikeAttr>()">,
        /*desc=*/"Value or container of values of BitSequenceType attribute"> {
    code storageType = "::mlir::bit::BitSequenceLikeAttr";
    code returnType = "::mlir::bit::BitSequenceLikeAttr";
    code convertFromStorage = "$_self";
}

#endif
