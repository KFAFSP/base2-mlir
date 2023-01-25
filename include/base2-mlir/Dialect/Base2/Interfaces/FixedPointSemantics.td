//===- FixedPointSemantics.td ------------------------------*- tablegen -*-===//
//
// This is the definitions file for the Base2 FixedPointSemantics interface.
//
//===----------------------------------------------------------------------===//

#ifndef BASE2_FIXEDPOINTSEMANTICS
#define BASE2_FIXEDPOINTSEMANTICS

include "mlir/IR/OpBase.td"

def Base2_FixedPointSemantics : TypeInterface<"FixedPointSemantics"> {
    let cppNamespace = "::mlir::base2";

    let description = [{
        Interface for a type that declares fixed-point semantics.
    }];

    let methods = [
        InterfaceMethod<
            /*desc=*/"Gets the associated integer type.",
            /*retTy=*/"IntegerType",
            /*methodName=*/"getIntegerType",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/"">,
        InterfaceMethod<
            /*desc=*/"Gets the number of fractional bits.",
            /*retTy=*/"bit_width_t",
            /*methodName=*/"getFractionalBits",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/"return 0;">];

    code extraTraitClassDeclaration = [{
    public:
        /// Gets the Signedness.
        [[nodiscard]] Signedness getSignedness() const
        {
            return base2::getSignedness(
                $_type.getIntegerType().getSignedness());
        }
        /// Determines whether these numbers are signless.
        [[nodiscard]] bool isSignless() const
        {
            return $_type.getSignedness() == Signedness::Signless;
        }
        /// Determines whether these numbers are signed.
        [[nodiscard]] bool isSigned() const
        {
            return $_type.getSignedness() == Signedness::Signed;
        }
        /// Determines whether these numbers are unsigned.
        [[nodiscard]] bool isUnsigned() const
        {
            return $_type.getSignedness() == Signedness::Unsigned;
        }

        /// Gets the total number of bits.
        [[nodiscard]] bit_width_t getBitWidth() const
        {
            return $_type.getIntegerType().getWidth();
        }
        /// Gets the number of integer bits.
        [[nodiscard]] bit_width_t getIntegerBits() const
        {
            return $_type.getBitWidth() - $_type.getFractionalBits();
        }
    }];

    code extraClassDeclaration = [{
    public:
        /// Gets the Signedness.
        [[nodiscard]] Signedness getSignedness() const
        {
            return base2::getSignedness(getIntegerType().getSignedness());
        }
        /// Determines whether these numbers are signless.
        [[nodiscard]] bool isSignless() const
        {
            return getSignedness() == Signedness::Signless;
        }
        /// Determines whether these numbers are signed.
        [[nodiscard]] bool isSigned() const
        {
            return getSignedness() == Signedness::Signed;
        }
        /// Determines whether these numbers are unsigned.
        [[nodiscard]] bool isUnsigned() const
        {
            return getSignedness() == Signedness::Unsigned;
        }

        /// Gets the total number of bits.
        [[nodiscard]] bit_width_t getBitWidth() const
        {
            return getIntegerType().getWidth();
        }
        /// Gets the number of integer bits.
        [[nodiscard]] bit_width_t getIntegerBits() const
        {
            return getBitWidth() - getFractionalBits();
        }

        //===--------------------------------------------------------------===//
        // BitSequence factories
        //===--------------------------------------------------------------===//

        /// Obtains the BitSequence representing zero.
        BitSequence getZero() const
        {
            return BitSequence::zeros(getBitWidth());
        }
        /// Obtains the BitSequence representing the smallest value, if any.
        [[nodiscard]] bit_result getMin() const
        {
            switch (getSignedness()) {
            case Signedness::Signless: return std::nullopt;
            case Signedness::Signed:
                return llvm::APInt::getSignedMinValue(getBitWidth());
            case Signedness::Unsigned:
                return llvm::APInt::getMinValue(getBitWidth());
            }
        }
        /// Obtains the BitSequence representing the largest value, if any.
        [[nodiscard]] bit_result getMax() const
        {
            switch (getSignedness()) {
            case Signedness::Signless: return std::nullopt;
            case Signedness::Signed:
                return llvm::APInt::getSignedMaxValue(getBitWidth());
            case Signedness::Unsigned:
                return llvm::APInt::getMaxValue(getBitWidth());
            }
        }

        //===--------------------------------------------------------------===//
        // Built-in
        //===--------------------------------------------------------------===//

        /// Gets the signless integer type with the same bit-width.
        [[nodiscard]] IntegerType getBitType() const
        {
            return IntegerType::get(getContext(), getBitWidth());
        }
        /// Gets the canonical built-in type with the same semantics, if any.
        [[nodiscard]] IntegerType getBuiltinType() const
        {
            if (getFractionalBits()) return IntegerType{};
            return getIntegerType();
        }

        //===--------------------------------------------------------------===//
        // Factories
        //===--------------------------------------------------------------===//

        /// Builds a canonical FixedPointSemantics type.
        ///
        /// @pre    `integerType`
        /// @pre    `fractionalBits <= integerType.getWidth()`
        [[nodiscard]] static FixedPointSemantics get(
            IntegerType integerType,
            bit_width_t fractionalBits = 0);
        /// Builds a canonical FixedPointSemantics type.
        ///
        /// @pre    `ctx`
        /// @pre    `integerBits + fractionalBits <= IntegerType::kMaxWidth`
        [[nodiscard]] static FixedPointSemantics get(
            MLIRContext *ctx,
            Signedness signedness,
            bit_width_t integerBits,
            bit_width_t fractionalBits = 0)
        {
            return get(
                IntegerType::get(
                    ctx,
                    integerBits + fractionalBits,
                    getBuiltinSignedness(signedness)),
                fractionalBits);
        }

        //===--------------------------------------------------------------===//
        // Formatting
        //===--------------------------------------------------------------===//

        /// Parses @p str as an abbreviated fixed-point semantics type.
        ///
        /// The abbreviated format is as follows:
        ///
        /// @code{.unparsed}
        /// fixed-sema      ::= (signedness-spec)? `i` [0-9]+ (`_` [0-9]+)?
        /// signedness-spec ::= `s` | `u`
        /// @endcode
        ///
        /// @pre    `ctx`
        [[nodiscard]] static FixedPointSemantics
        parse(MLIRContext* ctx, StringRef str);
        /// Prints abbreviated fixed-point semantics to @p out .
        ///
        /// See parse(MLIRContext*, StringRef) for a format description.
        void print(llvm::raw_ostream &out) const;

        //===--------------------------------------------------------------===//
        // Comparison
        //===--------------------------------------------------------------===//

        /// Determines whether @p rhs is equivalent to this.
        ///
        /// @pre    `rhs`
        [[nodiscard]] bool equals(FixedPointSemantics rhs) const
        {
            assert(rhs);

            return getIntegerType() == rhs.getIntegerType()
                && getFractionalBits() == rhs.getFractionalBits();
        }

        /// Determines whether @p rhs is a subset of this.
        ///
        /// @pre    `rhs`
        [[nodiscard]] bool isSupersetOf(FixedPointSemantics rhs) const;
    }];
}

//===----------------------------------------------------------------------===//
// FixedPointLike
//===----------------------------------------------------------------------===//

def Base2_FixedPointLike
        : TypeConstraint<
            /*predicate=*/CPred<"$_self.isa<::mlir::base2::FixedPointLikeType>()">,
            /*summary=*/"Type or container of FixedPointSemantics",
            /*cppClassName=*/"::mlir::base2::FixedPointLikeType">;

#endif
