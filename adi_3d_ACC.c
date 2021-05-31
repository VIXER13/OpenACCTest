#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#define MAX(a,b) ((a)>(b)?(a):(b))

double get_time () {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.;
}

#define N 64+2
double eps = 0.;
double A[N][N][N];

void init() {
#pragma acc parallel loop present(A)
    for(size_t k = 0; k < N; ++k)
        for(size_t i = 0; i < N; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i][j][k] = !i || i == N-1 || !j || j == N-1 || !k || k == N-1 ? 0. : (4. + i + j + k);
}

void relax() {
#pragma acc parallel present(A)
{
    for(size_t i = 1; i < N-1; ++i)
#pragma acc loop
        for(size_t j = 1; j < N-1; ++j)
            for(size_t k = 1; k < N-1; ++k)
                A[i][j][k] = 0.5 * (A[i-1][j][k] + A[i+1][j][k]);
}

#pragma acc parallel loop present(A)
    for(size_t i = 1; i < N-1; ++i)
        for(size_t j = 1; j < N-1; ++j)
            for(size_t k = 1; k < N-1; ++k)
                A[i][j][k] = 0.5 * (A[i][j-1][k] + A[i][j+1][k]);

#pragma acc parallel loop present(A) reduction(max : eps)
    for(size_t i = 1; i < N-1; ++i)
        for(size_t j = 1; j < N-1; ++j)
            for(size_t k = 1; k < N-1; ++k) {
                double e = A[i][j][k];
                A[i][j][k] = 0.5 * (A[i][j][k-1] + A[i][j][k+1]);
                e -= A[i][j][k];
                eps = MAX(eps, MAX(e, -e));
            }
}

void verify()
{
        double s = 0;
#pragma acc update self(A)
        for(size_t i = 0; i < N; ++i)
            for(size_t j = 0; j < N; ++j)
                for(size_t k = 0; k < N; ++k)
                    s += A[i][j][k] * (i+1)*(j+1)*(k+1) / (N*N*N);
        printf("S = %f\n",s);
}

int main(int argc, char** argv) {
        static const double maxeps = 1e-8;
        static const size_t itmax = 100;
#pragma acc enter data create(A[N][N][N], eps)
        init();
        double time = get_time();
        for(size_t it = 0; it < itmax; ++it) {
            eps = 0.;
            relax();
#pragma acc update self(eps)
            printf("it=%4li   eps=%f\n", it, eps);
            if(eps < maxeps)
                break;
        }
        time = get_time() - time;
        printf("time = %lf\n", time);
        verify();
#pragma acc exit data delete(A, eps)
        return EXIT_SUCCESS;
}