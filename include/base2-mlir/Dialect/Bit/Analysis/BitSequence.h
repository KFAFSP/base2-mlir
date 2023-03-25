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
#include <limits>
#include <type_traits>
#include <utility>

namespace mlir::bit {

/// Concept requiring that @p T is a C++ BitSequenceType.
template<class T>
concept bit_sequence =
    std::is_unsigned_v<T> && std::numeric_limits<T>::radix == 2;

/// Type that can hold the size of a bit sequence.
using bit_width_t = unsigned int;
/// Maximum value of the bit_width_t type.
static constexpr auto max_bit_width = std::numeric_limits<bit_width_t>::max();

/// Compile-time constant indicating the length of @p BitSequence .
template<bit_sequence BitSequence>
static constexpr bit_width_t bit_sequence_length =
    std::numeric_limits<BitSequence>::digits;

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
///
/// @note   This class isn't `constexpr` because `llvm::APInt` isn't.
class [[nodiscard]] BitSequence {
public:
    /// Type that holds the size.
    using size_type = bit_width_t;
    /// The underlying storage type.
    using storage_type = llvm::APInt;
    /// Type that stores bits in memory.
    using word_type = storage_type::WordType;
    /// Number of bits per storage word.
    static constexpr size_type word_bits = bit_sequence_length<word_type>;

    //===------------------------------------------------------------------===//
    // Builder
    //===------------------------------------------------------------------===//

    /// Mutable builder type for constructing BitSequence instances.
    class Builder {
    public:
        /// Initializes an empty Builder.
        /*implicit*/ Builder()
                : m_size(0),
                  m_lsbCapacity(word_bits),
                  m_buffer(1, 0)
        {
            m_lsb = m_buffer.begin();
        }

        //===--------------------------------------------------------------===//
        // Mutators
        //===--------------------------------------------------------------===//

        /// Appends @p n lsb bits of @p bits to the LSB.
        template<bit_sequence Bits>
        void append(Bits bits, bit_width_t n = bit_sequence_length<Bits>)
        {
            // Extends the word buffer in the LSB direction.
            const auto growLsb = [&]() {
                assert(m_lsbCapacity == 0);

                // Attempt to use pre-allocated buffer space.
                if (m_lsb-- == m_buffer.begin()) {
                    // Allocate new buffer space.
                    m_lsb = m_buffer.insert(m_buffer.begin(), word_type(0));
                }

                // A whole word is now available at m_lsb.
                m_lsbCapacity = word_bits;
            };

            // Align the input bits to the MSB.
            constexpr auto bitsWidth = bit_sequence_length<Bits>;
            bits <<= bitsWidth - n;

            // Obtains i MSB bits and shifts them out.
            const auto takeMsb = [&](bit_width_t i) {
                assert(i <= word_bits);

                // Obtain the MSB and shift it out.
                const Bits result = bits >> (bitsWidth - i);
                bits <<= i;
                return word_type(result);
            };

            // Append bits in chunks.
            while (n > 0) {
                // Ensure there is space to add bits.
                if (m_lsbCapacity == 0) growLsb();

                // Put the biggest possible chunk from the input MSB into m_lsb.
                const auto takeN = std::min(n, m_lsbCapacity);
                *m_lsb <<= takeN;
                *m_lsb |= takeMsb(takeN);

                // Update the counters.
                n -= takeN;
                m_lsbCapacity -= takeN;
                m_size += takeN;
            }
        }

        /// Prepends @p n lsb bits of @p bits to the MSB.
        template<bit_sequence Bits>
        void prepend(Bits bits, bit_width_t n = bit_sequence_length<Bits>)
        {
            // Obtains i LSB bits and shifts them out.
            const auto takeLsb = [&](bit_width_t i) {
                assert(i <= word_bits);

                // Obtain the LSB and shift it out.
                const Bits mask = (Bits(1) << i) - 1;
                const Bits result = bits & mask;
                bits >>= i;
                return word_type(result);
            };

            // Determine where and how big the MSB is.
            auto msb = this->msb();
            bit_width_t msbSize =
                m_size + m_lsbCapacity - word_bits * (msb - m_lsb);

            // Handle overlapping case.
            if (msb == m_lsb) {
                // Put the biggest possible chunk from the input LSB into m_lsb.
                const auto takeN = std::min(n, m_lsbCapacity);
                *m_lsb |= takeLsb(takeN) << (word_bits - m_lsbCapacity);

                // Update m_lsb's counters.
                n -= takeN;
                m_lsbCapacity -= takeN;
                m_size += takeN;

                // Proceed with a filled msb.
                msbSize = word_bits;
            }

            // Extends the word buffer in the MSB direction.
            const auto growMsb = [&]() {
                assert(msbSize == word_bits);

                // Attempt to use pre-allocated buffer space.
                if (++msb == m_buffer.end()) {
                    // Allocate new buffer space.
                    const auto lsbIdx = m_lsb - m_buffer.begin();
                    msb = &m_buffer.emplace_back(0);

                    // Restore m_lsb to the same index.
                    m_lsb = m_buffer.begin() + lsbIdx;
                }

                // A whole word is now available at msb.
                msbSize = 0;
            };

            // Prepend bits in chunks.
            while (n > 0) {
                // Ensure there is space to add bits.
                if (msbSize == word_bits) growMsb();

                // Put the biggest possible chunk from the input LSB into msb.
                const auto takeN = std::min(n, word_bits - msbSize);
                *msb |= takeLsb(takeN) << msbSize;

                // Update the counters.
                n -= takeN;
                msbSize += takeN;
                m_size += takeN;
            }
        }

        //===--------------------------------------------------------------===//
        // Container interface
        //===--------------------------------------------------------------===//

        /// Determines whether this sequence is empty.
        bool empty() const { return size() == 0; }
        /// Gets the current sequence length.
        bit_width_t size() const { return m_size; }

        /// Reserve capacity for @p capacity additional LSB bits.
        void reserve_lsb(bit_width_t capacity)
        {
            // Handle existing LSB capacity.
            if (capacity <= m_lsbCapacity) return;
            capacity -= m_lsbCapacity;

            // Reserve leading LSB words.
            const auto wantedLead = (capacity + word_bits - 1) / word_bits;
            const auto currentLead = m_lsb - m_buffer.begin();
            if (wantedLead <= currentLead) return;
            m_buffer.insert(m_buffer.begin(), wantedLead - currentLead, 0);

            // Restore m_lsb to the same word.
            m_lsb = m_buffer.begin() + wantedLead;
        }
        /// Reserve capacity for @p capacity additional MSB bits.
        void reserve_msb(bit_width_t capacity)
        {
            auto msb = this->msb();
            if (msb == m_lsb) {
                if (capacity <= m_lsbCapacity) return;
                capacity -= m_lsbCapacity;
            }

            // Reserve trailing MSB words.
            const auto wantedTrail = (capacity + word_bits - 1) / word_bits;
            const auto currentTrail = m_buffer.end() - msb - 1;
            if (wantedTrail <= currentTrail) return;
            const auto lsbIdx = m_lsb - m_buffer.begin();
            m_buffer.insert(m_buffer.end(), wantedTrail - currentTrail, 0);

            // Restore m_lsb to the same index.
            m_lsb = m_buffer.begin() + lsbIdx;
        }

        /// Truncates the sequence to @p newSize .
        ///
        /// @pre    `newSize() <= size()`
        /// @post   `size() == newSize`
        void truncate(bit_width_t newSize)
        {
            assert(newSize <= size());
            auto n = size() - newSize;

            // Determine where and how big the MSB is.
            auto msb = this->msb();
            bit_width_t msbSize =
                size() + m_lsbCapacity - word_bits * (msb - m_lsb);

            // Drop bits from MSB.
            while (msb != m_lsb) {
                const auto dropN = std::min(n, msbSize);
                *msb-- &= word_type(-1) >> (word_bits - msbSize + dropN);

                n -= dropN;
                m_size -= dropN;
                msbSize = word_bits;
            }

            // Drop bits from LSB.
            assert(n <= (word_bits - m_lsbCapacity));
            *m_lsb &= word_type(-1) >> (m_lsbCapacity + n);
            m_lsbCapacity += n;
            m_size -= n;
        }

        /// Resizes the sequence by removing or zero-inserting MSB bits.
        ///
        /// @post   `size() == newSize`
        void resize(bit_width_t newSize)
        {
            if (size() >= newSize)
                truncate(newSize);
            else
                prepend(0U, newSize - size());
        }

        //===--------------------------------------------------------------===//
        // Build methods
        //===--------------------------------------------------------------===//

        /// Converts the contained value to an llvm::APInt.
        llvm::APInt toUInt()
        {
            const auto msb = this->msb();

            if (m_lsbCapacity != 0) {
                // Distance of the bit shift.
                const auto offset = word_bits - m_lsbCapacity;

                // Shift all bits towards the LSB.
                for (auto it = m_lsb; it != msb;) {
                    *it |= *(it + 1) << offset;
                    *++it >>= m_lsbCapacity;
                }
                m_lsbCapacity = 0;
            }

            // Build the APInt.
            return llvm::APInt(
                m_size,
                llvm::ArrayRef<word_type>(m_lsb, msb + 1));
        }

        /// Converts the contained value to a BitSequence.
        BitSequence toBitSequence() { return toUInt(); }

    private:
        word_type* msb()
        {
            return m_lsb + ((m_size + m_lsbCapacity - 1) / word_bits);
        }

        size_type m_size;
        word_type* m_lsb;
        size_type m_lsbCapacity;
        llvm::SmallVector<word_type> m_buffer;
    };

    //===------------------------------------------------------------------===//
    // Initialization
    //===------------------------------------------------------------------===//

    /// Initializes an empty BitSequence.
    /*implicit*/ BitSequence() : m_impl(0, 0, false) {}

    // NOTE: clang-format doesn't handle requires clauses well.
    // clang-format off

    /// Initializes a BitSequence from @p n lsb bits of @p bits .
    template<bit_sequence Bits>
    requires(bit_sequence_length<Bits> <= word_bits)
    /*implicit*/ BitSequence(
            Bits bits,
            bit_width_t n = bit_sequence_length<Bits>)
            : m_impl(n, bits, false)
    {}

    // clang-format on

    /// Initializes a BitSequence from @p apint .
    /*implicit*/ BitSequence(storage_type apint) : m_impl(std::move(apint)) {}
    /// Initializes a bit sequence from @p apfloat .
    /*implicit*/ BitSequence(const llvm::APFloat &apfloat)
            : m_impl(apfloat.bitcastToAPInt())
    {}

    /// Initializes a BitSequence of @p width filled with @p fillBit .
    static BitSequence fill(bit_width_t width, bool fillBit)
    {
        return fillBit ? llvm::APInt::getAllOnes(width)
                       : llvm::APInt::getZero(width);
    }
    /// Initializes a BitSequence of @p width zeros.
    static BitSequence zeros(bit_width_t width) { return fill(width, false); }
    /// Initializes a BitSequence of @p width ones.
    static BitSequence ones(bit_width_t width) { return fill(width, true); }
    /// Initializes a BitSequence of @p width copying @p lsb .
    static BitSequence fromLSB(bit_width_t width, word_type lsb)
    {
        return llvm::APInt(width, lsb, false);
    }

    //===------------------------------------------------------------------===//
    // Accessors
    //===------------------------------------------------------------------===//

    /// Obtains the contained bit sequence as an unsigned llvm::APInt.
    [[nodiscard]] const storage_type &asUInt() const { return m_impl; }

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

    //===------------------------------------------------------------------===//
    // Logic operations
    //===------------------------------------------------------------------===//

    /// Computes the bitwise complement.
    BitSequence logicCmpl() const { return ~asUInt(); }
    /// Computes the bitwise logical and with @p rhs .
    BitSequence logicAnd(const BitSequence &rhs) const
    {
        return asUInt() & rhs.asUInt();
    }
    /// Computes the bitwise logical or with @p rhs .
    BitSequence logicOr(const BitSequence &rhs) const
    {
        return asUInt() | rhs.asUInt();
    }
    /// Computes the bitwise exclusive or with @p rhs .
    BitSequence logicXor(const BitSequence &rhs) const
    {
        return asUInt() ^ rhs.asUInt();
    }

    //===------------------------------------------------------------------===//
    // Shifting operations
    //===------------------------------------------------------------------===//

    /// Performs a logical left shift.
    BitSequence logicShl(bit_width_t amount) const
    {
        return asUInt().shl(amount);
    }
    /// Performs a logical left shift with funnel @p rhs .
    BitSequence funnelShl(const BitSequence &rhs, bit_width_t amount) const
    {
        auto result = asUInt().concat(rhs.asUInt()).shl(amount);
        result.lshrInPlace(size());
        return result.trunc(size());
    }

    /// Performs a logical right shift.
    BitSequence logicShr(bit_width_t amount) const
    {
        return asUInt().lshr(amount);
    }
    /// Performs a logical right shift with funnel @p lhs.
    BitSequence funnelShr(const BitSequence &lhs, bit_width_t amount) const
    {
        auto result = lhs.asUInt().concat(asUInt());
        result.lshrInPlace(amount);
        return result.trunc(size());
    }

    //===------------------------------------------------------------------===//
    // Scanning operations
    //===------------------------------------------------------------------===//

    /// Counts the number of 1 bits.
    [[nodiscard]] bit_width_t countOnes() const
    {
        return asUInt().countPopulation();
    }
    /// Counts the number of leading 0 bits.
    [[nodiscard]] bit_width_t countLeadingZeros() const
    {
        return asUInt().countLeadingZeros();
    }
    /// Counts the number of trailing 0 bits.
    [[nodiscard]] bit_width_t countTrailingZeros() const
    {
        return asUInt().countTrailingZeros();
    }

    //===------------------------------------------------------------------===//
    // Container interface
    //===------------------------------------------------------------------===//

    /// @copydoc Determines whether this sequence is empty.
    [[nodiscard]] bool empty() const { return size() == 0; }
    /// @copydoc Gets the number of bits.
    [[nodiscard]] size_type size() const { return asUInt().getBitWidth(); }

    /// Determines whether this sequence is just zero bits.
    [[nodiscard]] bool isZeros() const { return asUInt().isZero(); }
    /// Determines whether this sequence is just one bits.
    [[nodiscard]] bool isOnes() const { return asUInt().isAllOnes(); }

    //===------------------------------------------------------------------===//
    // Equality comparison
    //===------------------------------------------------------------------===//

    /// Determines whether @p other is the same bit sequence.
    [[nodiscard]] bool operator==(const storage_type &other) const
    {
        return asUInt() == other;
    }
    /// @copydoc operator==(const storage_type &)
    [[nodiscard]] bool operator==(const BitSequence &other) const
    {
        return *this == other.asUInt();
    }

    /// Computes a hash value for @p bits .
    [[nodiscard]] friend llvm::hash_code hash_value(const BitSequence &bits)
    {
        return llvm::hash_value(bits.asUInt());
    }

    //===------------------------------------------------------------------===//
    // Serialization
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
    operator<<(llvm::raw_ostream &out, const BitSequence &bits)
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
    friend AsmPrinter &operator<<(AsmPrinter &printer, const BitSequence &bits)
    {
        printer.getStream() << '"' << bits << '"';
        return printer;
    }

    //===------------------------------------------------------------------===//
    // Deserialization
    //===------------------------------------------------------------------===//

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

private:
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

} // namespace mlir::bit

namespace mlir {

template<>
struct FieldParser<bit::BitSequence> {
    static FailureOr<bit::BitSequence> parse(AsmParser &parser);
};

} // namespace mlir
