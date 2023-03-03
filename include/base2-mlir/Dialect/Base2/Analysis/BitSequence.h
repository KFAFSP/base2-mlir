/// Declares the compile-time bit sequence literal types.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <bit>
#include <cassert>
#include <cstdint>
#include <utility>

namespace mlir::base2 {

/// Type that can hold the size of a bit sequence.
using bit_width_t = unsigned int;

/// Type that represents the exponent of a fixed-point number.
using exponent_t = std::int32_t;

//===----------------------------------------------------------------------===//
// BitSequenceBase
//===----------------------------------------------------------------------===//

/// Base class for bit sequences.
///
/// The bit sequence types used by Base2 are simply opaque wrappers aroud llvm's
/// APInt type. This base class hides this implementation detail and the API of
/// the APInt type, enforcing the unopinionated character of bit sequences.
///
/// Additionally, the base class provides printing capabilities for writing the
/// contained bit sequence to a character stream in an invariant manner.
///
/// @note   This type isn't `constexpr` because llvm::APInt isn't.
class BitSequenceBase {
public:
    /// Type that holds the size.
    using size_type = bit_width_t;
    /// The underlying storage type.
    using storage_type = llvm::APInt;
    /// Type that stores bits in memory.
    using word_type = storage_type::WordType;

    /// Initializes an empty bit sequence.
    /*implicit*/ BitSequenceBase() : m_impl(0, 0, false) {}
    /// Initializes a bit sequence from @p storage .
    /*implicit*/ BitSequenceBase(storage_type storage)
            : m_impl(std::move(storage))
    {}
    /// Initializes a bit sequence from @p apfloat .
    /*implicit*/ BitSequenceBase(const llvm::APFloat &apfloat)
            : m_impl(apfloat.bitcastToAPInt())
    {}

    /// Obtains the contained bit sequence as an unsigned llvm::APInt.
    [[nodiscard]] const storage_type &asUInt() const { return m_impl; }

    /// @copydoc Determines whether this sequence is empty.
    [[nodiscard]] bool empty() const { return size() == 0; }
    /// @copydoc Gets the number of bits.
    [[nodiscard]] size_type size() const { return asUInt().getBitWidth(); }

    /// Determines whether this sequence is just zero bits.
    [[nodiscard]] bool isZeros() const { return asUInt().isZero(); }

    //===------------------------------------------------------------------===//
    // Equality comparison
    //===------------------------------------------------------------------===//

    /// Determines whether @p other is the same bit sequence.
    [[nodiscard]] bool operator==(const storage_type &other) const
    {
        return asUInt() == other;
    }
    /// @copydoc operator==(const storage_type &)
    [[nodiscard]] bool operator==(const BitSequenceBase &other) const
    {
        return *this == other.asUInt();
    }

    /// Computes a hash value for @p bits .
    [[nodiscard]] friend llvm::hash_code hash_value(const BitSequenceBase &bits)
    {
        return llvm::hash_value(bits.asUInt());
    }

    //===------------------------------------------------------------------===//
    // Formatting
    //===------------------------------------------------------------------===//

    /// Copies the bytes of this bit sequence into @p result .
    ///
    /// Similar to print(llvm::raw_ostream &), the minimum number of bytes
    /// required to fit all bits will be emitted. Inserted padding bits are 0.
    ///
    /// @pre    `endian == std::endian::big || endian == std::endian::little`
    void getBytes(
        llvm::SmallVectorImpl<std::uint8_t> &result,
        std::endian endian = std::endian::native) const;

    /// Writes a bit sequence to @p out as a hexadecimal bitstring literal.
    ///
    /// The format emitted by this printer is given by:
    ///
    /// @verbatim
    /// bit-sequence ::= `0x` [_0-9a-fA-F]* (`/` [0-9]+)?
    /// @endverbatim
    ///
    /// The printer only emits whole bytes (pairs of nibbles / hex digits), but
    /// omits all leading zero bytes. The truncation specifier `/` will be
    /// present in the result if the number of written bits does not equal the
    /// length of this bit sequence.
    void print(llvm::raw_ostream &out) const;

    /// Writes @p bits to @p out as a hexadecimal bitstring literal.
    ///
    /// See write(llvm::raw_ostream &) for more information.
    friend llvm::raw_ostream &
    operator<<(llvm::raw_ostream &out, const BitSequenceBase &bits)
    {
        bits.print(out);
        return out;
    }
    /// Writes @p bits to @p printer as a quoted hexadecimal bitstring literal.
    ///
    /// This overload wraps the literal in `"` quotes to ensure it can be
    /// unambiguously recovered using an associated mlir::FieldParser.
    ///
    /// See write(llvm::raw_ostream &) for more information.
    friend AsmPrinter &
    operator<<(AsmPrinter &printer, const BitSequenceBase &bits)
    {
        printer.getStream() << '"' << bits << '"';
        return printer;
    }

protected:
    /// Gets the stored words.
    ///
    /// @post   `!result.empty()`
    [[nodiscard]] llvm::ArrayRef<word_type> getWords() const
    {
        llvm::ArrayRef<word_type> result(
            m_impl.getRawData(),
            m_impl.getNumWords());

        assert(!result.empty());
        return result;
    }
    /// Gets the stored words, trimming al leading zero words, but not the LSB.
    ///
    /// @post   `!result.empty()`
    [[nodiscard]] llvm::ArrayRef<word_type> getActiveWords() const
    {
        llvm::ArrayRef<word_type> result(
            m_impl.getRawData(),
            m_impl.getActiveWords());

        assert(!result.empty());
        return result;
    }

    storage_type m_impl;
};

//===----------------------------------------------------------------------===//
// BitSequence
//===----------------------------------------------------------------------===//

/// Stores an immutable sequence of ordered bits.
///
/// A BitSequence is canonically ordered from MSB to LSB.
///
/// A BitSequence differs from an llvm::APInt in that:
///     - it is immutable,
///     - it does not prescribe how the bits are interpreted, and
///     - it does not support arithmetic operations.
///
/// Additionally, a BitSequence can be used as an MLIR field type, where it is
/// always represented using a string literal of the syntax:
///
/// @code{.unparsed}
/// bit-sequence ::= `"` bit-literal (trunc-spec)? `"`
/// bit-literal  ::= `0b` [_01]* | `0x` [_0-9a-fA-F]*
/// trunc-spec   ::= `/` [0-9]+
/// @endcode
class [[nodiscard]] BitSequence : public BitSequenceBase {
public:
    /// Mutable builder type for the BitSequence type.
    class Builder : public BitSequenceBase {
    public:
        using BitSequenceBase::BitSequenceBase;

        /// Ensure the capacity is not less than @p capacity .
        void reserve(size_type capacity)
        {
            // TODO: Consider a pre-allocatable implementation type.
            std::ignore = capacity;
        }

        /// Truncate to the @p newSize bits of LSB.
        ///
        /// @pre    `newSize <= size()`
        /// @post   `size() == newSize`
        void truncate(size_type newSize)
        {
            assert(newSize <= size());

            m_impl = m_impl.trunc(newSize);

            assert(size() == newSize);
        }
        /// Truncate or prepend zeros until @p newSize is reached.
        ///
        /// @post   `size() == newSize`
        void resize(size_type newSize)
        {
            m_impl = m_impl.zextOrTrunc(newSize);

            assert(size() == newSize);
        }

        /// Appends @p lsb .
        void append(const llvm::APInt &lsb) { m_impl = m_impl.concat(lsb); }
        /// Appends @p bit .
        void append(bool bit) { append(llvm::APInt(1, bit, false)); }
        /// Prepends @p msb .
        void prepend(llvm::APInt msb)
        {
            append(msb);
            m_impl = m_impl.rotr(msb.getBitWidth());
        }
        /// Prepends @p bit .
        void prepend(bool bit) { prepend(llvm::APInt(1, bit, false)); }

        /// Builds a BitSequence.
        BitSequence toBitSequence() const { return asUInt(); }
    };

    using BitSequenceBase::BitSequenceBase;

    /// Initializes a single-bit BitSequence.
    /*implicit*/ BitSequence(bool value)
            : BitSequenceBase(llvm::APInt(1, value, false))
    {}

    /// Initializes a BitSequence of @p width filled with @p fillBit .
    static BitSequence fill(bit_width_t width, bool fillBit)
    {
        return fillBit ? llvm::APInt::getAllOnes(width)
                       : llvm::APInt::getZero(width);
    }
    /// Initializes a BitSequence of @p width zeros.
    static BitSequence zeros(bit_width_t width) { return fill(width, false); }
    /// Initializes a BitSequence of @p width copying @p lsb .
    static BitSequence fromLSB(bit_width_t width, word_type lsb)
    {
        return llvm::APInt(width, lsb, false);
    }

    /// Initializes a BitSequence from @p bytes .
    ///
    /// Performs the inverse operation of getBytes().
    ///
    /// @pre    `bytes.size() >= bitWidth * 8`
    /// @pre    `endian == std::endian::big || endian == std::endian::little`
    static BitSequence fromBytes(
        ArrayRef<std::uint8_t> bytes,
        bit_width_t bitWidth,
        std::endian endian = std::endian::native);
    /// Reads packed BitSequence values from @p bytes .
    ///
    /// Performs the inverse operation of getBytes().
    ///
    /// @pre    `endian == std::endian::big || endian == std::endian::little`
    static void fromBytes(
        llvm::SmallVectorImpl<BitSequence> &result,
        ArrayRef<std::uint8_t> bytes,
        bit_width_t bitWidth,
        std::endian endian = std::endian::native);

    /// Gets the @p width LSB bits.
    ///
    /// If @p width is larger than `size()`, the result MSB is padded with 0.
    BitSequence getLSB(size_type width) const
    {
        return asUInt().zextOrTrunc(width);
    }
    /// Gets the @p width MSB bits.
    ///
    /// If @p width is larger than `size()`, the result LSB is padded with 0.
    BitSequence getMSB(size_type width) const
    {
        if (width <= size()) {
            auto result = asUInt();
            result.lshrInPlace(size() - width);
            return result.trunc(width);
        }

        return asUInt().zext(width).shl(width - size());
    }
};

/// Copies the bytes of @p values into @p result .
///
/// If @p values are single-bit sequences, the bits will be densely-packed
/// into bytes in LSB to MSB order. Otherwise, acts the same as
/// getBytes(llvm::SmallVectorImpl<std::uint8_t> &, std::endian).
///
/// @pre    all @p values have the same size
/// @pre    `endian == std::endian::big || endian == std::endian::little`
void getBytes(
    llvm::SmallVectorImpl<std::uint8_t> &result,
    ArrayRef<BitSequence> values,
    std::endian endian = std::endian::native);

} // namespace mlir::base2

namespace mlir {

template<>
struct FieldParser<base2::BitSequence> {
    static FailureOr<base2::BitSequence> parse(AsmParser &parser);
};

} // namespace mlir
