#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>

#define MAX(a,b) ((a)>(b)?(a):(b))

static double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

static void init(double *const A, const size_t N) {
#pragma acc parallel loop independent present(A)
    for(size_t i = 0; i < N; ++i)
#pragma acc loop independent
            for(size_t j = 0; j < N; ++j)
                A[i*N + j] = !i || i == N-1 ||
                             !j || j == N-1 ? 0. : 1. + (i + j);
}

static double relax(double *const A, const size_t N) {
    double eps = 0;
#pragma acc parallel loop independent present(A) reduction(max : eps)
    for(size_t i = 1; i < N-1; ++i)
#pragma acc loop independent
            for(size_t j = (i + 0) % 2 + 1; j < N-1; j += 2) {
                double e = A[i*N + j];
                A[i*N + j] = 0.25 * (A[(i-1)*N + j] + A[i*N + j-1] +
                                     A[(i+1)*N + j] + A[i*N + j+1]);
                e -= A[i*N + j];
                eps = MAX(eps, MAX(e, -e));
            }

#pragma acc parallel loop independent present(A) reduction(max : eps)
    for(size_t i = 1; i < N-1; ++i)
#pragma acc loop independent
            for(size_t j = (i + 1) % 2 + 1; j < N-1; j += 2) {
                double e = A[i*N + j];
                A[i*N + j] = 0.25 * (A[(i-1)*N + j] + A[i*N + j-1] +
                                     A[(i+1)*N + j] + A[i*N + j+1]);
                e -= A[i*N + j];
                eps = MAX(eps, MAX(e, -e));
            }
    return eps;
}

static void verify(const double *const A, const size_t N) {
    double s = 0.;
#pragma acc update self(A[0:N*N])
    for(size_t i = 0; i < N; ++i)
        for(size_t j = 0; j < N; ++j)
            s += A[i*N + j] * (i + 1) * (j + 1) / (N * N);
    printf("S = %f\n", s);
}

int main() {
    static const uintmax_t itmax = 100;
    static const double maxeps = 1e-8;
    for(size_t N = 1024 + 2; N < 1024 * 8 + 3; N = (N-2) * 2 + 2) {
        printf("SIZE = %lu\n", N-2);
        double *const A = (double*)malloc(N * N * sizeof(double));
#pragma acc data create(A[0:N*N])
        {
            init(A, N);
            const double time = get_time();
            for(uintmax_t it = 0; it < itmax; ++it) {
                const double eps = relax(A, N);
                if (eps < maxeps)
                    break;
            }
#pragma acc wait
            printf("time = %lf\n", get_time() - time);
            verify(A, N);
            printf("\n");
        }
        free(A);
    }
    return EXIT_SUCCESS;
}