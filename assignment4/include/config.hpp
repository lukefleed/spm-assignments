#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <cstddef>

// Default record payload size
#ifndef RPAYLOAD
#define RPAYLOAD 64
#endif

// Performance tuning constants
constexpr size_t MIN_PARTITION_SIZE = 1000;
constexpr size_t CACHE_LINE_SIZE = 64;
constexpr int DEFAULT_NUM_THREADS = 4;
constexpr size_t DEFAULT_ARRAY_SIZE = 1000000;

// MPI communication tags
constexpr int MPI_TAG_DATA = 100;
constexpr int MPI_TAG_SIZE = 101;
constexpr int MPI_TAG_MERGE = 102;
constexpr int MPI_TAG_CONTROL = 103;

// FastFlow optimization
constexpr bool FF_BOUNDED = true;
constexpr size_t FF_BUFFER_SIZE = 1024;

// Debug configuration
#ifdef DEBUG
#include <iostream>
#define DEBUG_PRINT(x) std::cout << "[DEBUG] " << x << std::endl
#else
#define DEBUG_PRINT(x)
#endif

// Compiler optimization hints
#ifdef __GNUC__
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#define FORCE_INLINE __attribute__((always_inline)) inline
#else
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#define FORCE_INLINE inline
#endif

// Memory alignment for SIMD
#define ALIGN_SIZE 64
#ifdef __GNUC__
#define ALIGNED __attribute__((aligned(ALIGN_SIZE)))
#else
#define ALIGNED
#endif

#endif // CONFIG_HPP
