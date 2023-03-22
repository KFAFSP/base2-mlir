/// Implements the compile-time bit sequence literal types.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "base2-mlir/Dialect/Base2/Analysis/BitSequence.h"

#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::base2;

/// Type of the words in storage.
using word_type = llvm::APInt::WordType;
/// Size of the words in storage in bits.
static constexpr auto word_bits = bit_sequence_length<word_type>;
/// Size of the words in storage in bytes.
static constexpr auto word_bytes = word_bits / 8;
static_assert(
    word_bytes * 8 == word_bits,
    "word_type must be densely packed bytes");
/// Number of nibbles needed to represent a storage word.
static constexpr auto word_nibbles = (word_bits + 3) / 4;
/// Mask to extract sub-word bit indices.
static constexpr auto word_mask = word_bits - 1UL;

/// Swap the order of the bytes in @p word .
static void byteSwap(word_type &word)
{
    // View the native bytes.
    const auto view = MutableArrayRef<std::uint8_t>(
        reinterpret_cast<std::uint8_t*>(&word),
        word_bytes);

    // Use a standard algorithm.
    std::reverse(view.begin(), view.end());
}

/// Copies the bytes of @p word to @p result .
///
/// @pre    `Endian == std::endian::big || Endian == std::endian::little`
template<std::endian Endian>
static void
copyBytes(const word_type &word, SmallVectorImpl<std::uint8_t> &result)
{
    static_assert(Endian == std::endian::big || Endian == std::endian::little);
    assert(
        std::endian::native == std::endian::big
        || std::endian::native == std::endian::little);

    // View the native bytes.
    const auto view = ArrayRef<std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(&word),
        word_bytes);

    // Copy to the result.
    if constexpr (Endian == std::endian::native)
        result.append(view.begin(), view.end());
    else
        result.append(view.rbegin(), view.rend());
}

/// @copydoc copyBytes(const word_type &, SmallVectorImpl<std::uint8_t> &)
static void copyBytes(
    const word_type &word,
    SmallVectorImpl<std::uint8_t> &result,
    std::endian endian)
{
    switch (endian) {
    case std::endian::little:
        copyBytes<std::endian::little>(word, result);
        break;
    case std::endian::big: copyBytes<std::endian::big>(word, result); break;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcovered-switch-default"
    default: assert(false && "not implemented"); break;
#pragma GCC diagnostic pop
    }
}

/// Obtains the bytes of @p word in big-endian order.
static SmallVector<std::uint8_t, word_bytes> getBytesBE(const word_type &word)
{
    SmallVector<std::uint8_t, word_bytes> result;
    copyBytes<std::endian::big>(word, result);
    return result;
}

/// Attempts to consume a binary digit from @p str into @p result .
static bool consumeBinDigit(StringRef &str, BitSequence::Builder &result)
{
    if (str.empty()) return false;

    const auto digit = str.front() - '0';
    if (digit < 0 || digit > 1) return false;

    str = str.drop_front();
    result.append(digit == 1);
    return true;
}

/// Attempts to consume a hexadecimal digit from @p str into @p result .
static bool consumeHexDigit(StringRef &str, BitSequence::Builder &result)
{
    if (str.empty()) return false;

    const auto ch = str.front();
    auto digit = ch - '0';
    if (digit < 0) return false;
    if (digit > 9) {
        digit = (ch | 0b00100000) - 'a' + 10;
        if (digit < 10 || digit > 15) return false;
    }

    str = str.drop_front();
    result.append(static_cast<unsigned>(digit), 4);
    return true;
}

//===----------------------------------------------------------------------===//
// BitSequence
//===----------------------------------------------------------------------===//

void BitSequence::getBytes(
    llvm::SmallVectorImpl<std::uint8_t> &result,
    std::endian endian) const
{
    const auto impl = [&](auto it, const auto end) {
        assert(it != end && "APInt always has 1 word");

        // Copy the MSB word, truncated to nearest bytes.
        if (const auto remBits = size() & word_mask) {
            // Skip all MSB bytes that are not defined.
            const auto bytes = getBytesBE(*it++);
            ArrayRef<std::uint8_t> activeBytes(bytes);
            activeBytes = activeBytes.drop_front((word_bits - remBits) / 8);

            // Perform the byte swap.
            if (endian != std::endian::big)
                result.append(activeBytes.rbegin(), activeBytes.rend());
            else
                result.append(activeBytes.begin(), activeBytes.end());
        }

        // Copy the LSB words, without truncation.
        while (it != end) copyBytes(*it++, result, endian);
    };

    // Words are organized from least to most significant.
    const auto words_le = getWords();
    switch (endian) {
    case std::endian::little: impl(words_le.begin(), words_le.end()); break;
    case std::endian::big: impl(words_le.rbegin(), words_le.rend()); break;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcovered-switch-default"
    default: assert(false && "not implemented"); break;
#pragma GCC diagnostic pop
    }
}

void BitSequence::print(llvm::raw_ostream &out) const
{
    out << "0x";

    // Words are organized from least to most significant.
    const auto words_le = getActiveWords();
    auto it = words_le.rbegin();
    const auto end = words_le.rend();
    assert(it != end && "APInt always has 1 word");

    std::size_t written = 0;

    // Print the MSB word, truncated to the active bytes.
    {
        // Skip all MSB bytes that are zero.
        const auto bytes = getBytesBE(*it);
        auto activeBytes = ArrayRef<uint8_t>(bytes);
        while (activeBytes.size() > 1 && activeBytes.front() == 0)
            activeBytes = activeBytes.drop_front();

        SmallString<word_nibbles> buffer;
        llvm::toHex(activeBytes, false, buffer);
        out << buffer;
        written += buffer.size();
    }

    // Print the LSB words, without truncation.
    while (++it != end) {
        const auto bytes = getBytesBE(*it);
        SmallString<word_nibbles> buffer;
        llvm::toHex(bytes, false, buffer);
        out << buffer;
        written += buffer.size();
    }

    // Write a truncation indicator if necessary.
    if (size() != (written * 4)) out << "/" << size();
}

BitSequence BitSequence::fromBytes(
    ArrayRef<std::uint8_t> bytes,
    bit_width_t bitWidth,
    std::endian endian)
{
    assert(bytes.size() * 8 >= bitWidth);
    assert(endian == std::endian::big || endian == std::endian::little);
    assert(
        std::endian::native == std::endian::big
        || std::endian::native == std::endian::little);

    // Quick exit if empty.
    if (bytes.empty()) return BitSequence();

    // Calculate the number of words in the input.
    const auto numInWords = bytes.size() / word_bytes;

    // View the input words.
    const auto inWords = ArrayRef<word_type>(
        reinterpret_cast<const word_type*>(bytes.data()),
        numInWords);

    // Calculate the number of words in the output.
    const auto numOutWords = (bytes.size() + word_bytes - 1) / word_bytes;

    // Copy the bytes into words.
    SmallVector<word_type> words;
    words.reserve(numOutWords);

    // Words are always stored in little-endian order.
    word_type* rem;
    if (endian == std::endian::little) {
        words.append(inWords.begin(), inWords.end());
        rem = &words.emplace_back(0);
    } else {
        rem = &words.emplace_back(0);
        words.append(inWords.rbegin(), inWords.rend());
    }
    MutableArrayRef<word_type> outWords = words;

    // The remainder of bytes spill into an extra word.
    bytes = bytes.drop_front(numInWords * word_bytes);
    if (!bytes.empty()) {
        const auto remBytes = MutableArrayRef<std::uint8_t>(
            reinterpret_cast<std::uint8_t*>(rem),
            word_bytes);

        if (endian == std::endian::little)
            for (auto [i, b] : llvm::enumerate(bytes)) remBytes[i] = b;
        else
            for (auto [i, b] : llvm::enumerate(bytes))
                remBytes[word_bytes - bytes.size() + i] = b;
    } else {
        // We did not end up using the remainder word.
        if (endian == std::endian::little)
            outWords = outWords.drop_back(1);
        else
            outWords = outWords.drop_front(1);
    }

    // If the word bytes are not in native endian order, swap them.
    if (endian != std::endian::native)
        for (auto &word : outWords) byteSwap(word);

    // Build the result.
    return llvm::APInt(bitWidth, outWords);
}

void BitSequence::fromBytes(
    llvm::SmallVectorImpl<BitSequence> &result,
    ArrayRef<std::uint8_t> bytes,
    bit_width_t bitWidth,
    std::endian endian)
{
    // Handle bit packing case.
    if (bitWidth == 1) {
        auto it = bytes.begin();
        const auto end = bytes.end();
        unsigned available = 0;
        std::uint8_t accu = 0;
        while (true) {
            if (available == 0) {
                if (it == end) break;
                accu = *it++;
                available = 8;
            }

            result.emplace_back(static_cast<bool>(accu & 0x01));
            accu >>= 1;
            --available;
        }

        return;
    }

    // Handle byte packing case.
    const auto bytesPerValue = std::max(0U, (bitWidth + 7) / 8);
    assert((bytes.size() % bytesPerValue) == 0);
    while (!bytes.empty()) {
        result.push_back(BitSequence::fromBytes(
            bytes.take_front(bytesPerValue),
            bitWidth,
            endian));
        bytes = bytes.drop_front(bytesPerValue);
    }
}

//===----------------------------------------------------------------------===//
// FieldParser<base2::BitSequence>
//===----------------------------------------------------------------------===//

FailureOr<BitSequence> mlir::FieldParser<BitSequence>::parse(AsmParser &parser)
{
    std::string str;
    if (parser.parseString(&str)) return failure();

    StringRef window(str);
    const auto emitError = [&]() {
        return parser.emitError(llvm::SMLoc::getFromPointer(
            parser.getCurrentLocation().getPointer() - window.size() - 2));
    };

    BitSequence::Builder builder;
    int state = 0;
    while (!window.empty()) {
        switch (state) {
        case 0:
            if (window.consume_front_insensitive("0b")) {
                builder.reserve_lsb(window.size() * 4);
                state = 1;
                break;
            }
            if (window.consume_front_insensitive("0x")) {
                builder.reserve_lsb(window.size() * 4);
                state = 2;
                break;
            }
            return emitError() << "expected `0b` or `0x`";

        case 1:
            if (window.consume_front("_")) break;
            if (consumeBinDigit(window, builder)) break;
            if (window.consume_front("/")) {
                state = 3;
                break;
            }
            return emitError() << "expected `_`, `0`, `1` or `/`";

        case 2:
            if (window.consume_front("_")) break;
            if (consumeHexDigit(window, builder)) break;
            if (window.consume_front("/")) {
                state = 3;
                break;
            }
            return emitError()
                   << "expected `_`, `0`-`9`, `a`-`f`, `A-F` or `/`";

        case 3:
            unsigned len;
            if (!window.consumeInteger(10, len)) {
                builder.resize(len);
                state = 4;
                break;
            }
            return emitError() << "expected `0`-`9`";

        case 4: return emitError() << "expected `\"`";
        }
    }

    if (state == 3) return emitError() << "expected `0`-`9`";

    return builder.toBitSequence();
}

void base2::getBytes(
    llvm::SmallVectorImpl<std::uint8_t> &result,
    ArrayRef<BitSequence> values,
    std::endian endian)
{
    // Quick exit on empty.
    if (values.empty()) return;

    // Handle bit packing case.
    const auto bitWidth = values.front().size();
    if (bitWidth == 1) {
        result.reserve(result.size() + (values.size() + 7) / 8);

        std::uint8_t accu = 0;
        unsigned used = 0;
        for (const auto &value : values) {
            assert(value.size() == 1);

            if (used == 8) {
                result.push_back(accu);
                accu = used = 0;
            }

            accu |= (value.asUInt().getZExtValue() << used);
            ++used;
        }

        if (used) result.push_back(accu);
        return;
    }

    // Handle byte packing case.
    const auto bytesPerValue = std::max(0U, (bitWidth + 7) / 8);
    result.reserve(result.size() + values.size() * bytesPerValue);
    for (const auto &value : values) {
        assert(value.size() == values.front().size());
        value.getBytes(result, endian);
    }
}
