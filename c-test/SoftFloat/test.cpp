#include <iomanip>
#include <iostream>

extern "C" int64_t _mlir_ciface_add_caller(
    int64_t lhs,
    int64_t rhs,
    int8_t expBits,
    int8_t fracBits,
    int32_t expBias,
    bool hasRounding,
    bool hasNan,
    bool hasOne,
    bool hasSubnorm,
    int8_t sign);

extern "C" int64_t _mlir_ciface_sub_caller(
    int64_t lhs,
    int64_t rhs,
    int8_t expBits,
    int8_t fracBits,
    int32_t expBias,
    bool hasRounding,
    bool hasNan,
    bool hasOne,
    bool hasSubnorm,
    int8_t sign);

extern "C" int64_t _mlir_ciface_mul_caller(
    int64_t lhs,
    int64_t rhs,
    int8_t expBits,
    int8_t fracBits,
    int32_t expBias,
    bool hasRounding,
    bool hasNan,
    bool hasOne,
    bool hasSubnorm,
    int8_t sign);

extern "C" int64_t _mlir_ciface_divg_caller(
    int64_t lhs,
    int64_t rhs,
    int8_t expBits,
    int8_t fracBits,
    int32_t expBias,
    bool hasRounding,
    bool hasNan,
    bool hasOne,
    bool hasSubnorm,
    int8_t sign);

extern "C" int64_t _mlir_ciface_divsrt_caller(
    int64_t lhs,
    int64_t rhs,
    int8_t expBits,
    int8_t fracBits,
    int32_t expBias,
    bool hasRounding,
    bool hasNan,
    bool hasOne,
    bool hasSubnorm,
    int8_t sign);

extern "C" bool _mlir_ciface_eq_caller(
    int64_t lhs,
    int64_t rhs,
    int8_t expBits,
    int8_t fracBits,
    int32_t expBias,
    bool hasRounding,
    bool hasNan,
    bool hasOne,
    bool hasSubnorm,
    int8_t sign);

extern "C" bool _mlir_ciface_le_caller(
    int64_t lhs,
    int64_t rhs,
    int8_t expBits,
    int8_t fracBits,
    int32_t expBias,
    bool hasRounding,
    bool hasNan,
    bool hasOne,
    bool hasSubnorm,
    int8_t sign);

extern "C" bool _mlir_ciface_lt_caller(
    int64_t lhs,
    int64_t rhs,
    int8_t expBits,
    int8_t fracBits,
    int32_t expBias,
    bool hasRounding,
    bool hasNan,
    bool hasOne,
    bool hasSubnorm,
    int8_t sign);

extern "C" bool _mlir_ciface_ge_caller(
    int64_t lhs,
    int64_t rhs,
    int8_t expBits,
    int8_t fracBits,
    int32_t expBias,
    bool hasRounding,
    bool hasNan,
    bool hasOne,
    bool hasSubnorm,
    int8_t sign);

extern "C" bool _mlir_ciface_gt_caller(
    int64_t lhs,
    int64_t rhs,
    int8_t expBits,
    int8_t fracBits,
    int32_t expBias,
    bool hasRounding,
    bool hasNan,
    bool hasOne,
    bool hasSubnorm,
    int8_t sign);

extern "C" bool _mlir_ciface_ltgt_caller(
    int64_t lhs,
    int64_t rhs,
    int8_t expBits,
    int8_t fracBits,
    int32_t expBias,
    bool hasRounding,
    bool hasNan,
    bool hasOne,
    bool hasSubnorm,
    int8_t sign);

extern "C" bool _mlir_ciface_nan_caller(
    int64_t in,
    int8_t expBits,
    int8_t fracBits,
    int32_t expBias,
    bool hasRounding,
    bool hasNan,
    bool hasOne,
    bool hasSubnorm,
    int8_t sign);

extern "C" int64_t _mlir_ciface_cast_caller(
    int64_t in,
    int8_t expBits,
    int8_t fracBits,
    int32_t expBias,
    bool hasRounding,
    bool hasNan,
    bool hasOne,
    bool hasSubnorm,
    int8_t sign,
    int8_t out_expBits,
    int8_t out_fracBits,
    int32_t out_expBias,
    bool out_hasRounding,
    bool out_hasNan,
    bool out_hasOne,
    bool out_hasSubnorm,
    int8_t out_sign);

extern "C" int64_t _mlir_ciface_castfloat_caller(
    double in,
    int8_t expBits,
    int8_t fracBits,
    int32_t expBias,
    bool hasRounding,
    bool hasNan,
    bool hasOne,
    bool hasSubnorm,
    int8_t sign);

extern "C" double _mlir_ciface_casttofloat_caller(
    int64_t in,
    int8_t expBits,
    int8_t fracBits,
    int32_t expBias,
    bool hasRounding,
    bool hasNan,
    bool hasOne,
    bool hasSubnorm,
    int8_t sign);

int main()
{
    int64_t one = 0x3ff0000000000000;
    int64_t half = 0x3fe0000000000000;
    int64_t nan = 0x7ff4000000000000;
    int8_t expBits = 11;
    int8_t fracBits = 52;
    int32_t expBias = -1023;
    bool hasRounding = true;
    bool hasNan = true;
    bool hasOne = true;
    bool hasSubnorm = true;
    int8_t sign = 0;

    int64_t lhs = one;
    int64_t rhs = half;

    std::cout << "Arithmetic operations test:" << std::endl;
    std::cout << "lhs: " << std::hex << lhs << std::endl;
    std::cout << "rhs: " << std::hex << rhs << std::endl;

    int64_t result = _mlir_ciface_add_caller(
        lhs,
        rhs,
        expBits,
        fracBits,
        expBias,
        hasRounding,
        hasNan,
        hasOne,
        hasSubnorm,
        sign);

    std::cout << "add: " << std::hex << result << std::endl;

    result = _mlir_ciface_sub_caller(
        lhs,
        rhs,
        expBits,
        fracBits,
        expBias,
        hasRounding,
        hasNan,
        hasOne,
        hasSubnorm,
        sign);

    std::cout << "sub: " << std::hex << result << std::endl;

    result = _mlir_ciface_mul_caller(
        lhs,
        rhs,
        expBits,
        fracBits,
        expBias,
        hasRounding,
        hasNan,
        hasOne,
        hasSubnorm,
        sign);

    std::cout << "mul: " << std::hex << result << std::endl;

    result = _mlir_ciface_divg_caller(
        lhs,
        rhs,
        expBits,
        fracBits,
        expBias,
        hasRounding,
        hasNan,
        hasOne,
        hasSubnorm,
        sign);

    std::cout << "divg: " << std::hex << result << std::endl;

    result = _mlir_ciface_divg_caller(
        lhs,
        rhs,
        expBits,
        fracBits,
        expBias,
        hasRounding,
        hasNan,
        hasOne,
        hasSubnorm,
        sign);

    std::cout << "divsrt: " << std::hex << result << std::endl;

    std::cout << std::endl;

    lhs = one;
    rhs = one;

    std::cout << "Comparison operations test:" << std::endl;
    std::cout << "lhs: " << std::hex << lhs << std::endl;
    std::cout << "rhs: " << std::hex << rhs << std::endl;

    bool comp = _mlir_ciface_eq_caller(
        lhs,
        rhs,
        expBits,
        fracBits,
        expBias,
        hasRounding,
        hasNan,
        hasOne,
        hasSubnorm,
        sign);

    std::cout << "eq: " << comp << std::endl;

    comp = _mlir_ciface_le_caller(
        lhs,
        rhs,
        expBits,
        fracBits,
        expBias,
        hasRounding,
        hasNan,
        hasOne,
        hasSubnorm,
        sign);

    std::cout << "le: " << comp << std::endl;

    comp = _mlir_ciface_lt_caller(
        lhs,
        rhs,
        expBits,
        fracBits,
        expBias,
        hasRounding,
        hasNan,
        hasOne,
        hasSubnorm,
        sign);

    std::cout << "lt: " << comp << std::endl;

    comp = _mlir_ciface_ge_caller(
        lhs,
        rhs,
        expBits,
        fracBits,
        expBias,
        hasRounding,
        hasNan,
        hasOne,
        hasSubnorm,
        sign);

    std::cout << "ge: " << comp << std::endl;

    comp = _mlir_ciface_gt_caller(
        lhs,
        rhs,
        expBits,
        fracBits,
        expBias,
        hasRounding,
        hasNan,
        hasOne,
        hasSubnorm,
        sign);

    std::cout << "gt: " << comp << std::endl;

    comp = _mlir_ciface_ltgt_caller(
        lhs,
        rhs,
        expBits,
        fracBits,
        expBias,
        hasRounding,
        hasNan,
        hasOne,
        hasSubnorm,
        sign);

    std::cout << "ltgt: " << comp << std::endl;

    int64_t in = nan;
    std::cout << "in: " << std::hex << in << std::endl;
    comp = _mlir_ciface_nan_caller(
        in,
        expBits,
        fracBits,
        expBias,
        hasRounding,
        hasNan,
        hasOne,
        hasSubnorm,
        sign);

    std::cout << "nan: " << comp << std::endl;
    std::cout << std::endl;

    in = one;
    int8_t out_expBits = 15;
    int8_t out_fracBits = 48;
    int32_t out_expBias = -16383;
    bool out_hasRounding = true;
    bool out_hasNan = true;
    bool out_hasOne = true;
    bool out_hasSubnorm = true;
    int8_t out_sign = 0;
    std::cout << "Cast operations test:" << std::endl;

    std::cout << "1. Cast sfloat number:" << std::endl;
    std::cout << "in: " << std::hex << in << std::endl;

    int64_t out = _mlir_ciface_cast_caller(
        in,
        expBits,
        fracBits,
        expBias,
        hasRounding,
        hasNan,
        hasOne,
        hasSubnorm,
        sign,
        out_expBits,
        out_fracBits,
        out_expBias,
        out_hasRounding,
        out_hasNan,
        out_hasOne,
        out_hasSubnorm,
        out_sign);

    std::cout << "cast: " << std::hex << out << std::endl;

    double float_in = 1.0;
    std::cout << "2. Cast f64 to sfloat number:" << std::endl;
    std::cout << "float_in: " << float_in << std::endl;

    out = _mlir_ciface_castfloat_caller(
        float_in,
        expBits,
        fracBits,
        expBias,
        hasRounding,
        hasNan,
        hasOne,
        hasSubnorm,
        sign);

    std::cout << "floatcast: " << std::hex << out << std::endl;

    in = half;
    std::cout << "3. Cast sfloat to f64 number:" << std::endl;
    std::cout << "in: " << std::hex << in << std::endl;

    double float_out = _mlir_ciface_casttofloat_caller(
        in,
        expBits,
        fracBits,
        expBias,
        hasRounding,
        hasNan,
        hasOne,
        hasSubnorm,
        sign);

    std::cout << "float_out: " << float_out << std::endl;

    return 0;
}
