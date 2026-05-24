// matmul.cpp — single-file matmul optimization task.
//
// optimize ONLY optimized_matmul(). gold_matmul() is the reference: it stays
// untouched and is run every time to (a) give the baseline runtime and (b)
// verify optimized_matmul() still produces the right matrix. The program prints
// the speedup and whether the optimized result still matches the reference.
//
// Build: g++ -O3 -std=c++17 -march=native -fopenmp matmul.cpp -o /tmp/matmul && /tmp/matmul

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>

static constexpr int N = 512;

// ===========================================================================
// GOLD REFERENCE — DO NOT MODIFY. This defines "correct" and the baseline.
// ===========================================================================
void gold_matmul(const double* A, const double* B, double* C) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double s = 0.0;
            for (int k = 0; k < N; ++k)
                s += A[i * N + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

// ===========================================================================
// OPTIMIZE THIS. Start: identical to gold_matmul. Make it as fast as you can
// while keeping the result equal to gold (the program checks this each run).
// ===========================================================================
void optimized_matmul(const double* A, const double* B, double* C) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double s = 0.0;
            for (int k = 0; k < N; ++k)
                s += A[i * N + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

// ===========================================================================
// Harness — do not modify; this is what the benchmark reports.
// ===========================================================================
static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

int main() {
    double* A  = (double*)malloc(N * N * sizeof(double));
    double* B  = (double*)malloc(N * N * sizeof(double));
    double* Cg = (double*)calloc(N * N, sizeof(double));
    double* Co = (double*)calloc(N * N, sizeof(double));

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < N * N; ++i) { A[i] = dist(rng); B[i] = dist(rng); }

    double t0 = now_ms();
    gold_matmul(A, B, Cg);
    double gold_ms = now_ms() - t0;

    double t1 = now_ms();
    optimized_matmul(A, B, Co);
    double opt_ms = now_ms() - t1;

    // Direct comparison against the reference matrix (relative tolerance).
    double max_diff = 0.0, max_ref = 0.0;
    for (int i = 0; i < N * N; ++i) {
        max_diff = std::fmax(max_diff, std::fabs(Cg[i] - Co[i]));
        max_ref  = std::fmax(max_ref,  std::fabs(Cg[i]));
    }
    double rel_err = max_diff / (max_ref + 1e-15);
    bool correct = rel_err < 1e-6;
    double speedup = opt_ms > 0 ? gold_ms / opt_ms : 0.0;

    std::printf("gold_ms:  %.3f\n", gold_ms);
    std::printf("opt_ms:   %.3f\n", opt_ms);
    std::printf("speedup:  %.3f\n", speedup);
    std::printf("rel_err:  %.3e\n", rel_err);
    std::printf("correct:  %s\n", correct ? "yes" : "no");

    free(A); free(B); free(Cg); free(Co);
    return correct ? 0 : 1;
}
