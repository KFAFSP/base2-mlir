#include "mlir/ExecutionEngine/CRunnerUtils.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <random>

#define N         11
#define size_2    N* N
#define size_3    N* N* N

#define EXP_BITS  11
#define FRAC_BITS 52
#define EXP_BIAS  -1023

using namespace std;

// TODO: compare with std version using mean square error(MSE)

extern "C" int64_t _mlir_ciface_cast_float(
    double a,
    int8_t exp_bits,
    int8_t frac_bits,
    int32_t exp_bias);
extern "C" double _mlir_ciface_cast_to_float(
    int64_t a,
    int8_t exp_bits,
    int8_t frac_bits,
    int32_t exp_bias);

extern "C" void _mlir_ciface_test_sf(
    int8_t exp_bits,
    int8_t frac_bits,
    int32_t exp_bias,
    StridedMemRefType<int64_t, 3>* A,
    StridedMemRefType<int64_t, 3>* B,
    StridedMemRefType<int64_t, 2>* C);

extern "C" void _mlir_ciface_test(
    StridedMemRefType<double, 3>* A,
    StridedMemRefType<double, 3>* B,
    StridedMemRefType<double, 2>* C);

void rand_num_frac(int size, double* fa, int64_t* ia)
{
    double temp = 0.0;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-10.0, 10.0);
    for (int i = 0; i < size; i++) {
        temp = dis(gen);
        fa[i] = temp;
        ia[i] = _mlir_ciface_cast_float(temp, EXP_BITS, FRAC_BITS, EXP_BIAS);
    }
}

void rand_num_exp(int size, double* fa, int64_t* ia)
{
    double temp = 0.0;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned int> frac_dist(0, (1 << 30) - 1);
    std::uniform_int_distribution<int> exp_dist(-44, -15);
    for (int i = 0; i < size; i++) {
        temp = ldexp(frac_dist(gen), exp_dist(gen));
        // cout << temp << "  ";
        fa[i] = temp;
        ia[i] = _mlir_ciface_cast_float(temp, EXP_BITS, FRAC_BITS, EXP_BIAS);
    }
}

int main()
{
    srand((unsigned)time(NULL));

    // Data given to softfloat version kernel
    double f_a_data[size_3];
    double f_b_data[size_3];
    double f_c_data[size_2];

    // Data given to softfloat version kernel
    int64_t i_a_data[size_3];
    int64_t i_b_data[size_3];
    int64_t i_c_data[size_2];

    // random numbers
    rand_num_exp(size_3, f_a_data, i_a_data);
    rand_num_exp(size_3, f_b_data, i_b_data);

    // MemRef of standard version of kernel test
    StridedMemRefType<double, 3> f_a{
        f_a_data,
        f_a_data,
        0,
        {  N,  N, N},
        {121, 11, 1}
    };
    StridedMemRefType<double, 3> f_b{
        f_b_data,
        f_b_data,
        0,
        {  N,  N, N},
        {121, 11, 1}
    };
    StridedMemRefType<double, 2> f_c{
        f_c_data,
        f_c_data,
        0,
        { N, N},
        {11, 1}
    };

    // MemRef of softfloat version of kernel test
    StridedMemRefType<int64_t, 3> i_a{
        i_a_data,
        i_a_data,
        0,
        {  N,  N, N},
        {121, 11, 1}
    };
    StridedMemRefType<int64_t, 3> i_b{
        i_b_data,
        i_b_data,
        0,
        {  N,  N, N},
        {121, 11, 1}
    };
    StridedMemRefType<int64_t, 2> i_c{
        i_c_data,
        i_c_data,
        0,
        { N, N},
        {11, 1}
    };

    _mlir_ciface_test(&f_a, &f_b, &f_c);

    _mlir_ciface_test_sf(EXP_BITS, FRAC_BITS, EXP_BIAS, &i_a, &i_b, &i_c);

    ofstream output;
    string file_str = "/home/bi/Desktop/base2-mlir/c-test/matmul/result/result_"
                      + to_string(EXP_BITS) + "_" + to_string(FRAC_BITS)
                      + "_exp.csv";
    output.open(file_str, ios::out | ios::trunc);
    if (!output.is_open()) {
        cout << "Error opening file: " + file_str << endl;
        return 1;
    }
    output << "c_std"
           << ","
           << "c_sf" << endl;
    for (int i = 0; i < size_2; i++) {
        output << f_c.data[i] << ","
               << _mlir_ciface_cast_to_float(
                      i_c.data[i],
                      EXP_BITS,
                      FRAC_BITS,
                      EXP_BIAS)
               << endl;
    }

    return 0;
}
