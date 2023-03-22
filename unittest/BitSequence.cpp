#include "base2-mlir/Dialect/Base2/Analysis/BitSequence.h"

#include "llvm/ADT/StringExtras.h"

#include <doctest/doctest.h>
#include <iostream>

using namespace mlir::base2;

// clang-format off

TEST_CASE("BitSequence::Builder::append") {
    BitSequence::Builder builder;

    SUBCASE("in-place") {
        builder.append(0b101010U, 6);
        builder.append(0b1111U, 4);

        REQUIRE(builder.size() == 10);
        REQUIRE(builder.toUInt().getZExtValue() == 0b1010101111U);
    }

    SUBCASE("allocate") {
        builder.reserve_lsb(16 * 8);
        for (unsigned i = 0; i < 16; ++i) builder.append(i, 8);

        REQUIRE(builder.size() == 16 * 8);
        std::string str;
        llvm::raw_string_ostream temp(str);
        temp << builder.toBitSequence();
        REQUIRE(str == "0x0102030405060708090A0B0C0D0E0F/128");
    }
}

TEST_CASE("BitSequence::Builder::prepend") {
    BitSequence::Builder builder;

    SUBCASE("in-place") {
        builder.prepend(0b101010U, 6);
        builder.prepend(0b1111U, 4);

        REQUIRE(builder.size() == 10);
        REQUIRE(builder.toUInt().getZExtValue() == 0b1111101010U);
    }

    SUBCASE("allocate") {
        builder.reserve_msb(16 * 8);
        for (unsigned i = 0; i < 16; ++i) builder.prepend(i, 8);

        REQUIRE(builder.size() == 16 * 8);
        std::string str;
        llvm::raw_string_ostream temp(str);
        temp << builder.toBitSequence();
        REQUIRE(str == "0x0F0E0D0C0B0A09080706050403020100");
    }
}

TEST_CASE("BitSequence::Builder::resize") {
    BitSequence::Builder builder;
    builder.reserve_lsb(9 * 8);
        for (unsigned i = 0; i < 9; ++i)
            builder.append(i, 8);

    SUBCASE("truncate") {
        builder.resize(8 * 8);

        REQUIRE(builder.size() == 8 * 8);
        REQUIRE(builder.toUInt().getZExtValue() == 0x0102030405060708);
    }

    SUBCASE("extend") {
        builder.resize(10 * 8);

        REQUIRE(builder.size() == 10 * 8);
        std::string str;
        llvm::raw_string_ostream temp(str);
        temp << builder.toBitSequence();
        REQUIRE(str == "0x0102030405060708/80");
    }
}

TEST_CASE("BitSequence::getBytes") {
    SUBCASE("padding") {
        BitSequence bits(0x3U, 8);

        llvm::SmallVector<std::uint8_t> result;
        bits.getBytes(result);

        REQUIRE(result.size() == 1);
        REQUIRE(result[0] == 0x03);
    }

    SUBCASE("std::endian::little") {
        BitSequence bits(0x0102030405060708UL);

        llvm::SmallVector<std::uint8_t> result;
        bits.getBytes(result, std::endian::little);

        REQUIRE(result.size() == 8);
        for (int i = 0; i < 8; ++i)
            REQUIRE(result[i] == 8 - i);
    }

    SUBCASE("std::endian::big") {
        BitSequence bits(0x0102030405060708UL);

        llvm::SmallVector<std::uint8_t> result;
        bits.getBytes(result, std::endian::big);

        REQUIRE(result.size() == 8);
        for (int i = 0; i < 8; ++i)
            REQUIRE(result[i] == i + 1);
    }
}

TEST_CASE("getBytes (boolean)") {
    llvm::SmallVector<BitSequence> bits = {
        true, false, false, true, true, false, true, true,
        true, true };
    llvm::SmallVector<std::uint8_t> result;

    SUBCASE("std::endian::little") {
        getBytes(result, bits, std::endian::little);

        REQUIRE(result.size() == 2);
        REQUIRE(result[0] == 0b11011001);
        REQUIRE(result[1] == 0b00000011);
    }

    SUBCASE("std::endian::big") {
        getBytes(result, bits, std::endian::big);

        REQUIRE(result.size() == 2);
        REQUIRE(result[0] == 0b11011001);
        REQUIRE(result[1] == 0b00000011);
    }
}

TEST_CASE("getBytes (packed)") {
    llvm::SmallVector<BitSequence> bits = {
        BitSequence(0x0123U, 12),
        BitSequence(0x0456U, 12),
        BitSequence(0x0789U, 12)};
    llvm::SmallVector<std::uint8_t> result;

    SUBCASE("std::endian::little") {
        getBytes(result, bits, std::endian::little);

        REQUIRE(result.size() == 6);
        REQUIRE(result[0] == 0x23);
        REQUIRE(result[1] == 0x01);
        REQUIRE(result[2] == 0x56);
        REQUIRE(result[3] == 0x04);
        REQUIRE(result[4] == 0x89);
        REQUIRE(result[5] == 0x07);
    }

    SUBCASE("std::endian::big") {
        getBytes(result, bits, std::endian::big);

        REQUIRE(result.size() == 6);
        REQUIRE(result[0] == 0x01);
        REQUIRE(result[1] == 0x23);
        REQUIRE(result[2] == 0x04);
        REQUIRE(result[3] == 0x56);
        REQUIRE(result[4] == 0x07);
        REQUIRE(result[5] == 0x89);
    }
}

TEST_CASE("BitSequence::fromBytes (remainder)") {
    BitSequence::Builder builder;
    builder.reserve_lsb(3 * 8);
    for (unsigned i = 0; i < 3; ++i) builder.append(i, 8);

    const auto bits = builder.toBitSequence();
    llvm::SmallVector<std::uint8_t> result;

    SUBCASE("std::endian::little") {
        getBytes(result, bits, std::endian::little);
        const auto copy =
            BitSequence::fromBytes(result, 3 * 8, std::endian::little);

        REQUIRE(copy == bits);
    }

    SUBCASE("std::endian::big") {
        getBytes(result, bits, std::endian::big);
        const auto copy =
            BitSequence::fromBytes(result, 3 * 8, std::endian::big);

        REQUIRE(copy == bits);
    }
}

TEST_CASE("BitSequence::fromBytes (copy)") {
    BitSequence::Builder builder;
    builder.reserve_lsb(16 * 8);
    for (unsigned i = 0; i < 16; ++i) builder.append(i, 8);

    const auto bits = builder.toBitSequence();
    llvm::SmallVector<std::uint8_t> result;

    SUBCASE("std::endian::little") {
        getBytes(result, bits, std::endian::little);
        const auto copy =
            BitSequence::fromBytes(result, 16 * 8, std::endian::little);

        REQUIRE(copy == bits);
    }

    SUBCASE("std::endian::big") {
        getBytes(result, bits, std::endian::big);
        const auto copy =
            BitSequence::fromBytes(result, 16 * 8, std::endian::big);

        REQUIRE(copy == bits);
    }
}

TEST_CASE("BitSequence::fromBytes (boolean)") {
    llvm::SmallVector<BitSequence> bits = {
        true, false, false, true, true, false, true, true,
        true, true };
    llvm::SmallVector<std::uint8_t> result;

    SUBCASE("std::endian::little") {
        getBytes(result, bits, std::endian::little);

        llvm::SmallVector<BitSequence> copies;
        BitSequence::fromBytes(copies, result, 1, std::endian::little);

        REQUIRE(copies.size() == 16);
        for (std::size_t i = 0; i < bits.size(); ++i)
            REQUIRE(copies[i] == bits[i]);
    }

    SUBCASE("std::endian::big") {
        getBytes(result, bits, std::endian::big);

        llvm::SmallVector<BitSequence> copies;
        BitSequence::fromBytes(copies, result, 1, std::endian::big);

        REQUIRE(copies.size() == 16);
        for (std::size_t i = 0; i < bits.size(); ++i)
            REQUIRE(copies[i] == bits[i]);
    }
}

TEST_CASE("BitSequence::fromBytes (packed)") {
    llvm::SmallVector<BitSequence> bits = {
        BitSequence(0x0123U, 12),
        BitSequence(0x0456U, 12),
        BitSequence(0x0789U, 12)};
    llvm::SmallVector<std::uint8_t> result;

    SUBCASE("std::endian::little") {
        getBytes(result, bits, std::endian::little);

        llvm::SmallVector<BitSequence> copies;
        BitSequence::fromBytes(copies, result, 12, std::endian::little);

        REQUIRE(copies.size() == bits.size());
        for (std::size_t i = 0; i < copies.size(); ++i)
            REQUIRE(bits[i] == copies[i]);
    }

    SUBCASE("std::endian::big") {
        getBytes(result, bits, std::endian::big);

        llvm::SmallVector<BitSequence> copies;
        BitSequence::fromBytes(copies, result, 12, std::endian::big);

        REQUIRE(copies.size() == bits.size());
        for (std::size_t i = 0; i < copies.size(); ++i) {
            llvm::errs() << bits[i] << " == " << copies[i] << "\n";
            REQUIRE(bits[i] == copies[i]);
        }
    }
}
