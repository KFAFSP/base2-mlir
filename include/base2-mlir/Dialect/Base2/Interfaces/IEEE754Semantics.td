//===- IEEE754Semantics.td ---------------------------------*- tablegen -*-===//
//
// This is the definitions file for the Base2 IEEE754Semantics interface.
//
//===----------------------------------------------------------------------===//

#ifndef BASE2_IEEE754SEMANTICS
#define BASE2_IEEE754SEMANTICS

include "mlir/IR/OpBase.td"

def Base2_IEEE754Semantics : TypeInterface<"IEEE754Semantics"> {
    let cppNamespace = "::mlir::base2";

    let description = [{
        Interface for a type that declares IEEE-754 semantics.
    }];

    let methods = [
        InterfaceMethod<
            /*desc=*/"Gets the precision in bits.",
            /*retTy=*/"bit_width_t",
            /*methodName=*/"getPrecision",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/"">,
        InterfaceMethod<
            /*desc=*/"Gets the number of exponent bits.",
            /*retTy=*/"bit_width_t",
            /*methodName=*/"getExponentBits",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/"">,
        InterfaceMethod<
            /*desc=*/"Gets the exponent bias.",
            /*retTy=*/"exponent_t",
            /*methodName=*/"getBias",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/[{
                return ieee754_semantics_interface_defaults::getBias($_type);
            }]>];

    code extraTraitClassDeclaration = [{
    public:
        /// Gets the total number of bits.
        [[nodiscard]] bit_width_t getBitWidth() const
        {
            return $_type.getPrecision() + $_type.getExponentBits();
        }
        /// Gets the number of bits stored in the mantissa field.
        [[nodiscard]] bit_width_t getMantissaBits() const
        {
            return $_type.getPrecision() - 1;
        }

        /// Gets the minimum representable exponent.
        [[nodiscard]] exponent_t getMinExponent() const
        {
            return base2::getMinExponent($_type.getBias());
        }
        /// Gets the maximum representable exponent.
        [[nodiscard]] exponent_t getMaxExponent() const
        {
            return base2::getMaxExponent(
                $_type.getBias(),
                $_type.getExponentBits());
        }
    }];

    code extraClassDeclaration = [{
    public:
        /// Gets the total number of bits.
        [[nodiscard]] bit_width_t getBitWidth() const
        {
            return getPrecision() + getExponentBits();
        }
        /// Gets the number of bits stored in the mantissa field.
        [[nodiscard]] bit_width_t getMantissaBits() const
        {
            return getPrecision() - 1;
        }

        /// Gets the minimum representable exponent.
        [[nodiscard]] exponent_t getMinExponent() const
        {
            return base2::getMinExponent(getBias());
        }
        /// Gets the maximum representable exponent.
        [[nodiscard]] exponent_t getMaxExponent() const
        {
            return base2::getMaxExponent(getBias(), getExponentBits());
        }

        //===--------------------------------------------------------------===//
        // BitSequence factories
        //===--------------------------------------------------------------===//

        /// Obtains the BitSequence representing zero.
        BitSequence getZero(bool negative = false) const
        {
            return negative
                ? llvm::APInt::getOneBitSet(getBitWidth(), getBitWidth() - 1)
                : BitSequence::zeros(getBitWidth());
        }
        /// Obtains the BitSequence representing the smallest value.
        BitSequence getMin() const
        {
            llvm::APInt result(getBitWidth(), 0, false);
            result.setHighBits(getExponentBits() + 1);
            return result;
        }
        /// Obtains the BitSequence representing the largest value.
        BitSequence getMax() const
        {
            llvm::APInt result(getBitWidth(), 0, false);
            result.setHighBits(getExponentBits() + 1);
            result.clearSignBit();
            return result;
        }
        /// Obtains the BitSequence representing a NaN value.
        BitSequence getNaN(bool negative = false) const
        {
            auto result = llvm::APInt::getAllOnes(getBitWidth());
            if (!negative) result.clearSignBit();
            return result;
        }
        /// Obtains the BitSequence representing a NaN value.
        ///
        /// @pre    `!data.isZero()`
        /// @pre    `data.size() <= getMantissaBits()`
        BitSequence getNaN(BitSequence data, bool negative = false) const
        {
            assert(!data.isZeros());
            assert(data.size() <= getMantissaBits());

            llvm::APInt result(getBitWidth(), 0, false);
            result.setHighBits(getExponentBits() + 1);
            if (!negative) result.clearSignBit();
            result |= data.asUInt().zext(getMantissaBits());
            return result;
        }

        //===--------------------------------------------------------------===//
        // Built-in
        //===--------------------------------------------------------------===//

        /// Gets the signless integer type with the same bit-width.
        [[nodiscard]] IntegerType getBitType() const
        {
            return IntegerType::get(getContext(), getBitWidth());
        }

        /// Gets the APFloat::Semantics matching this, if any.
        [[nodiscard]] std::optional<APFloat::Semantics> toLLVMSemantics() const;
        /// Gets the built-in type for @p llvmSema .
        ///
        /// @pre    `ctx`
        [[nodiscard]] static FloatType
        getBuiltinType(MLIRContext *ctx, APFloat::Semantics llvmSema);
        /// Gets the canonical built-in type with the same semantics, if any.
        [[nodiscard]] FloatType getBuiltinType() const
        {
            if (const auto llvmSema = toLLVMSemantics())
                return getBuiltinType(getContext(), *llvmSema);

            return FloatType{};
        }

        //===--------------------------------------------------------------===//
        // Factories
        //===--------------------------------------------------------------===//

        /// Gets a canonical IEEE754Semantics type.
        ///
        /// @pre    `ctx`
        /// @pre    `precision >= 2`
        /// @pre    `exponentBits > 1 && exponentBits <= max_exponent_bits`
        /// @pre    `getBitWidth(bias) <=> exponentBits`
        [[nodiscard]] static IEEE754Semantics get(
            MLIRContext *ctx,
            bit_width_t precision,
            bit_width_t exponentBits,
            exponent_t bias);
        /// Gets a canonical IEEE754Semantics type.
        ///
        /// @pre    `ctx`
        /// @pre    `precision >= 2`
        /// @pre    `maxExponent >= minExponent`
        [[nodiscard]] static IEEE754Semantics get(
            MLIRContext *ctx,
            bit_width_t precision,
            exponent_t maxExponent,
            exponent_t minExponent)
        {
            const auto [exponentBits, bias] =
                getExponentParams(maxExponent, minExponent);
            return get(ctx, precision, exponentBits, bias);
        }
        /// Gets a canonical IEEE754Semantics type.
        ///
        /// @pre    `ctx`
        /// @pre    `precision >= 2`
        [[nodiscard]] static IEEE754Semantics get(
            MLIRContext *ctx,
            bit_width_t precision,
            exponent_t maxExponent)
        {
            return get(ctx, precision, exponent_t(1) - maxExponent);
        }

        //===--------------------------------------------------------------===//
        // Formatting
        //===--------------------------------------------------------------===//

        /// Parses @p str as an abbreviated IEEE-754 semantics type.
        ///
        /// The abbreviated format is as follows:
        ///
        /// @code{.unparsed}
        /// ieee754-sema ::= `f` [0-9]+ `_` [0-9]+ (`_` bias-spec)?
        /// bias-spec    ::= (`m`)? [0-9]+
        /// @endcode
        ///
        /// @pre    `ctx`
        [[nodiscard]] static IEEE754Semantics
        parse(MLIRContext* ctx, StringRef str);
        /// Prints abbreviated IEEE-754 semantics to @p out .
        ///
        /// See parse(MLIRContext*, StringRef) for a format description.
        void print(llvm::raw_ostream &out) const;

        //===--------------------------------------------------------------===//
        // Comparison
        //===--------------------------------------------------------------===//

        /// Determines whether @p rhs is equivalent to this.
        ///
        /// @pre    `rhs`
        [[nodiscard]] bool equals(IEEE754Semantics rhs) const
        {
            assert(rhs);

            return getPrecision() == rhs.getPrecision()
                && getExponentBits() == rhs.getExponentBits()
                && getBias() == rhs.getBias();
        }
        /// Determine whether @p rhs is equivale to this.
        ///
        /// @note   MLIR's interface constructor will cause a non-recoverable
        ///         template substitution error if we use a reference to an
        ///         incomplete in any context where implicit conversion may
        ///         happen, e.g., in this overload set.
        [[nodiscard]] bool equals(const llvm::fltSemantics *rhs) const;

        /// Determines whether @p rhs is a subset of this.
        ///
        /// @pre    `rhs`
        [[nodiscard]] bool isSupersetOf(IEEE754Semantics rhs) const
        {
            assert(rhs);

            return getPrecision() >= rhs.getPrecision()
                && getMaxExponent() >= rhs.getMaxExponent()
                && getMinExponent() <= rhs.getMinExponent();
        }
        /// Determine whether @p rhs is a subset of this.
        ///
        /// @note   MLIR's interface constructor will cause a non-recoverable
        ///         template substitution error if we use a reference to an
        ///         incomplete in any context where implicit conversion may
        ///         happen, e.g., in this overload set.
        [[nodiscard]] bool isSupersetOf(const llvm::fltSemantics *rhs) const;
    }];
}

//===----------------------------------------------------------------------===//
// IEEE754SemanticsLike
//===----------------------------------------------------------------------===//

def Base2_IEEE754SemanticsLike
        : TypeConstraint<
            /*predicate=*/CPred<"$_self.isa<::mlir::base2::IEEE754SemanticsLikeType>()">,
            /*summary=*/"Type or container of IEEE754Semantics",
            /*cppClassName=*/"::mlir::base2::IEEE754SemanticsLikeType">;

#endif
