#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>

using Matrix = std::vector<std::vector<double>>;

// Initializes an N x N matrix with random floating-point values
Matrix generateMatrix(int n) {
    Matrix mat(n, std::vector<double>(n));
    std::mt19937_64 rng(1337); // Fixed seed for reproducible benchmarks
    std::uniform_real_distribution<double> dist(1.0, 100.0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            mat[i][j] = dist(rng);
        }
    }
    return mat;
}

// Naive matrix multiplication: standard i -> j -> k loop order
// Induces frequent CPU cache misses due to strided column access in B
Matrix multiplyNaive(const Matrix& A, const Matrix& B, int n) {
    Matrix C(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

// Cache-friendly matrix multiplication: loop reordered to i -> k -> j
// Ensures continuous, sequential row-major reads from memory
Matrix multiplyCacheOptimized(const Matrix& A, const Matrix& B, int n) {
    Matrix C(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            double r = A[i][k];
            for (int j = 0; j < n; ++j) {
                C[i][j] += r * B[k][j];
            }
        }
    }
    return C;
}

void runBenchmark(int n) {
    std::cout << "Generating " << n << "x" << n << " matrices...\n";
    Matrix A = generateMatrix(n);
    Matrix B = generateMatrix(n);

    // Benchmark Naive (i-j-k)
    auto startNaive = std::chrono::high_resolution_clock::now();
    Matrix cNaive = multiplyNaive(A, B, n);
    auto endNaive = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> naiveDuration = endNaive - startNaive;

    // Benchmark Cache-Optimized (i-k-j)
    auto startOpt = std::chrono::high_resolution_clock::now();
    Matrix cOpt = multiplyCacheOptimized(A, B, n);
    auto endOpt = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> optDuration = endOpt - startOpt;

    double speedup = naiveDuration.count() / optDuration.count();

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Dimension: " << n << "x" << n << "\n";
    std::cout << "  - Naive (i-j-k)         : " << naiveDuration.count() << " ms\n";
    std::cout << "  - Cache-Friendly (i-k-j): " << optDuration.count() << " ms\n";
    std::cout << "  - Efficiency Speedup    : " << speedup << "x faster\n";
    std::cout << "----------------------------------------------------\n";
}

int main() {
    std::cout << "====================================================\n";
    std::cout << "   Matrix Benchmark Engine: Memory Locality Analysis \n";
    std::cout << "====================================================\n\n";

    std::vector<int> dimensions = {128, 256, 512};
    for (int dim : dimensions) {
        runBenchmark(dim);
    }

    return 0;
}
