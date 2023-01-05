#include "mlir/ExecutionEngine/CRunnerUtils.h"

#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>

#define N      11
#define size_2 N* N
#define size_3 N* N* N

using namespace std;

// TODO: compare with std version using mean square error(MSE)

extern "C" int64_t _mlir_ciface_cast_float(double a);

extern "C" void _mlir_ciface_kernel_sf(
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

double frand()
{
    int t = rand() % 2 + 1;
    double fp = pow(-1, t) * (double)rand() / RAND_MAX;
    return fp * 2.0;
}

void rand_num(int size, double* fa, int64_t* ia)
{
    double temp = 0.0;
    for (int i = 0; i < size; i++) {
        temp = frand();
        fa[i] = temp;
        ia[i] = _mlir_ciface_cast_float(temp);
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

    // random number for S/D/u in double and i64(sfloat)
    rand_num(size_2, f_S_data, i_S_data);
    rand_num(size_3, f_D_data, i_D_data);
    rand_num(size_3, f_u_data, i_u_data);

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

    time_t now = time(0);
    char buffer[32];
    strftime(buffer, 32, "%Y-%m-%d-%H-%M-%S", localtime(&now));
    string time_str(buffer);
    string file_std = time_str + "_std.csv";
    string file_sf = time_str + "_sf.csv";

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

    ofstream f_std;
    f_std.open(
        "/home/bi/Desktop/base2dialect/c-test/rescheduled/result/" + file_std,
        ios::out | ios::trunc);
    f_std << "v"
          << ","
          << "t"
          << ","
          << "r"
          << ","
          << "t0"
          << ","
          << "t1"
          << ","
          << "t2"
          << ","
          << "t3" << endl;
    for (int i = 0; i < size_3; i++) {
        f_std << f_v.data[i] << "," << f_t.data[i] << "," << f_r.data[i] << ","
              << f_t0.data[i] << "," << f_t1.data[i] << "," << f_t2.data[i]
              << "," << f_t3.data[i] << endl;
    }

    _mlir_ciface_kernel_sf(
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

    ofstream i_std;
    i_std.open(
        "/home/bi/Desktop/base2dialect/c-test/rescheduled/result/" + file_sf,
        ios::out | ios::trunc);
    i_std << "v"
          << ","
          << "t"
          << ","
          << "r"
          << ","
          << "t0"
          << ","
          << "t1"
          << ","
          << "t2"
          << ","
          << "t3" << endl;
    for (int i = 0; i < size_3; i++) {
        i_std << hex << i_v.data[i] << "," << hex << i_t.data[i] << "," << hex
              << i_r.data[i] << "," << hex << i_t0.data[i] << "," << hex
              << i_t1.data[i] << "," << hex << i_t2.data[i] << "," << hex
              << i_t3.data[i] << endl;
    }

    return 0;
}
