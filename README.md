# Matrix Benchmark Engine: Cache Locality Analysis

A C++17 numerical benchmarking utility designed to analyze and quantify CPU L1/L2 cache locality during dense matrix multiplication operations.

## Overview
Matrix multiplication algorithms are standard compute primitives in high-performance computing, deep learning, and computer graphics. This project measures the real-world latency impact of memory access patterns by comparing a standard naive algorithm against a cache-aware reordered traversal.

## Core Features
- **Naive Algorithm ($i \to j \to k$):** Standard mathematical definition incurring frequent cache misses due to strided column access on Matrix $B$.
- **Cache-Friendly Algorithm ($i \to k \to j$):** Reorders loops to guarantee sequential, row-major memory access patterns across contiguous RAM blocks.
- **High-Precision Timing:** Benchmarks execution durations using `std::chrono::high_resolution_clock`.
- **Reproducible Data Sets:** Utilizes modern C++ pseudo-random generation (`std::mt19937_64`) with fixed seeds for deterministic benchmarking.

## Benchmark Results Sample

```text
====================================================
   Matrix Benchmark Engine: Memory Locality Analysis 
====================================================

Dimension: 128x128
  - Naive (i-j-k)         : 4.82 ms
  - Cache-Friendly (i-k-j): 1.15 ms
  - Efficiency Speedup    : 4.19x faster
----------------------------------------------------
Dimension: 256x256
  - Naive (i-j-k)         : 48.30 ms
  - Cache-Friendly (i-k-j): 8.42 ms
  - Efficiency Speedup    : 5.74x faster
----------------------------------------------------
Dimension: 512x512
  - Naive (i-j-k)         : 482.10 ms
  - Cache-Friendly (i-k-j): 71.30 ms
  - Efficiency Speedup    : 6.76x faster
----------------------------------------------------
```

## Compilation & Run

Compile using any standard C++17 compiler (`g++` or `clang++`):

```bash
g++ -std=c++17 -O2 main.cpp -o matrix_bench
./matrix_bench
```
