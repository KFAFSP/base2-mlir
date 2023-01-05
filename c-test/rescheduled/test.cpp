#include "mlir/ExecutionEngine/CRunnerUtils.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <random>

#define N         11
#define size_2    N* N
#define size_3    N* N* N

#define EXP_BITS  11
#define FRAC_BITS 27
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

extern "C" void _mlir_ciface_kernel_sf(
    int8_t exp_bits,
    int8_t frac_bits,
    int32_t exp_bias,
    StridedMemRefType<int64_t, 2>* S,
    StridedMemRefType<int64_t, 3>* D,
    StridedMemRefType<int64_t, 3>* u,
    StridedMemRefType<int64_t, 3>* v,
    StridedMemRefType<int64_t, 3>* t,
    StridedMemRefType<int64_t, 3>* r,
    StridedMemRefType<int64_t, 3>* t0,
    StridedMemRefType<int64_t, 3>* t1,
    StridedMemRefType<int64_t, 3>* t2,
    StridedMemRefType<int64_t, 3>* t3);

extern "C" void _mlir_ciface_kernel_std(
    StridedMemRefType<double, 2>* S,
    StridedMemRefType<double, 3>* D,
    StridedMemRefType<double, 3>* u,
    StridedMemRefType<double, 3>* v,
    StridedMemRefType<double, 3>* t,
    StridedMemRefType<double, 3>* r,
    StridedMemRefType<double, 3>* t0,
    StridedMemRefType<double, 3>* t1,
    StridedMemRefType<double, 3>* t2,
    StridedMemRefType<double, 3>* t3);

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
        fa[i] = temp;
        ia[i] = _mlir_ciface_cast_float(temp, EXP_BITS, FRAC_BITS, EXP_BIAS);
    }
}

int main()
{
    srand((unsigned)time(NULL));

    // Data given to softfloat version kernel
    double f_S_data[size_2];
    double f_D_data[size_3];
    double f_u_data[size_3];
    double f_v_data[size_3];
    double f_t_data[size_3];
    double f_r_data[size_3];
    double f_t0_data[size_3];
    double f_t1_data[size_3];
    double f_t2_data[size_3];
    double f_t3_data[size_3];

    // Data given to softfloat version kernel

    int64_t i_S_data[size_2];
    int64_t i_D_data[size_3];
    int64_t i_u_data[size_3];
    int64_t i_v_data[size_3];
    int64_t i_t_data[size_3];
    int64_t i_r_data[size_3];
    int64_t i_t0_data[size_3];
    int64_t i_t1_data[size_3];
    int64_t i_t2_data[size_3];
    int64_t i_t3_data[size_3];

    // random numbers
    rand_num_exp(size_2, f_S_data, i_S_data);
    rand_num_exp(size_3, f_D_data, i_D_data);
    rand_num_exp(size_3, f_u_data, i_u_data);

    // MemRef of standard version of kernel test
    StridedMemRefType<double, 2> f_S{
        f_S_data,
        f_S_data,
        0,
        { N, N},
        {10, 1}
    };
    StridedMemRefType<double, 3> f_D{
        f_D_data,
        f_D_data,
        0,
        {  N,  N, N},
        {100, 10, 1}
    };
    StridedMemRefType<double, 3> f_u{
        f_u_data,
        f_u_data,
        0,
        {  N,  N, N},
        {100, 10, 1}
    };
    StridedMemRefType<double, 3> f_v{
        f_v_data,
        f_v_data,
        0,
        {  N,  N, N},
        {100, 10, 1}
    };
    StridedMemRefType<double, 3> f_t{
        f_t_data,
        f_t_data,
        0,
        {  N,  N, N},
        {100, 10, 1}
    };
    StridedMemRefType<double, 3> f_r{
        f_r_data,
        f_r_data,
        0,
        {  N,  N, N},
        {100, 10, 1}
    };
    StridedMemRefType<double, 3> f_t0{
        f_t0_data,
        f_t0_data,
        0,
        {  N,  N, N},
        {100, 10, 1}
    };
    StridedMemRefType<double, 3> f_t1{
        f_t1_data,
        f_t1_data,
        0,
        {  N,  N, N},
        {100, 10, 1}
    };
    StridedMemRefType<double, 3> f_t2{
        f_t2_data,
        f_t2_data,
        0,
        {  N,  N, N},
        {100, 10, 1}
    };
    StridedMemRefType<double, 3> f_t3{
        f_t3_data,
        f_t3_data,
        0,
        {  N,  N, N},
        {100, 10, 1}
    };

    // MemRef of softfloat version of kernel test
    StridedMemRefType<int64_t, 2> i_S{
        i_S_data,
        i_S_data,
        0,
        { N, N},
        {10, 1}
    };
    StridedMemRefType<int64_t, 3> i_D{
        i_D_data,
        i_D_data,
        0,
        {  N,  N, N},
        {100, 10, 1}
    };
    StridedMemRefType<int64_t, 3> i_u{
        i_u_data,
        i_u_data,
        0,
        {  N,  N, N},
        {100, 10, 1}
    };
    StridedMemRefType<int64_t, 3> i_v{
        i_v_data,
        i_v_data,
        0,
        {  N,  N, N},
        {100, 10, 1}
    };
    StridedMemRefType<int64_t, 3> i_t{
        i_t_data,
        i_t_data,
        0,
        {  N,  N, N},
        {100, 10, 1}
    };
    StridedMemRefType<int64_t, 3> i_r{
        i_r_data,
        i_r_data,
        0,
        {  N,  N, N},
        {100, 10, 1}
    };
    StridedMemRefType<int64_t, 3> i_t0{
        i_t0_data,
        i_t0_data,
        0,
        {  N,  N, N},
        {100, 10, 1}
    };
    StridedMemRefType<int64_t, 3> i_t1{
        i_t1_data,
        i_t1_data,
        0,
        {  N,  N, N},
        {100, 10, 1}
    };
    StridedMemRefType<int64_t, 3> i_t2{
        i_t2_data,
        i_t2_data,
        0,
        {  N,  N, N},
        {100, 10, 1}
    };
    StridedMemRefType<int64_t, 3> i_t3{
        i_t3_data,
        i_t3_data,
        0,
        {  N,  N, N},
        {100, 10, 1}
    };

    _mlir_ciface_kernel_std(
        &f_S,
        &f_D,
        &f_u,
        &f_v,
        &f_t,
        &f_r,
        &f_t0,
        &f_t1,
        &f_t2,
        &f_t3);

    _mlir_ciface_kernel_sf(
        EXP_BITS,
        FRAC_BITS,
        EXP_BIAS,
        &i_S,
        &i_D,
        &i_u,
        &i_v,
        &i_t,
        &i_r,
        &i_t0,
        &i_t1,
        &i_t2,
        &i_t3);

    ofstream output;
    string file_str =
        "/home/bi/Desktop/base2-mlir/c-test/rescheduled/result/result_"
        + to_string(EXP_BITS) + "_" + to_string(FRAC_BITS) + "_exp.csv";
    output.open(file_str, ios::out | ios::trunc);
    if (!output.is_open()) {
        cout << "Error opening file: " + file_str << endl;
        return 1;
    }
    output << "v_std"
           << ","
           << "v_sf" << endl;
    for (int i = 0; i < size_3; i++) {
        output << f_v.data[i] << ","
               << _mlir_ciface_cast_to_float(
                      i_v.data[i],
                      EXP_BITS,
                      FRAC_BITS,
                      EXP_BIAS)
               << endl;
    }

    return 0;
}
