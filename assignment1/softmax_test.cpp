/**
 * @file softmax_test.cpp
 * @brief Benchmark and validation framework for softmax implementations
 *
 * This test framework evaluates three implementations of the softmax function:
 * 1. Plain (baseline serial implementation)
 * 2. Auto-vectorized (compiler-optimized implementation)
 * 3. AVX (manually vectorized using AVX/AVX2/AVX512 intrinsics)
 *
 * The framework performs:
 * - Performance benchmarks across multiple input sizes
 * - Correctness verification of outputs
 * - Thread scaling analysis
 * - Numerical stability testing
 *
 * Results are recorded in CSV format for further analysis.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem> // For directory creation
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map> // For std::map
#include <memory>
#include <new>     // For std::align_val_t
#include <numeric> // For std::accumulate
#include <omp.h>   // For OpenMP threading
#include <random>
#include <set> // For std::set
#include <vector>

namespace fs = std::filesystem;

/**
 * @struct BenchmarkResult
 * @brief Container for benchmark timing results of all implementations
 *
 * Stores execution times for the three softmax implementations for a specific
 * input size.
 */
struct BenchmarkResult {
  size_t size;
  double t_plain;
  double t_auto;
  double t_avx;
};

/**
 * @brief Get the result file name based on compile-time configuration
 *
 * Constructs a file path based on the current compilation mode:
 * - PARALLEL flag controls whether parallel execution was enabled
 * - USE_AVX512 flag indicates whether AVX512 instructions were used
 *
 * @return std::string File path for results
 */
std::string getResultFileName() {
  std::string name = "results/performance/results";

#if PARALLEL == 0
  name += "_noparallel";
#else
  name += "_parallel";
#endif

#if USE_AVX512 == 0
  name += "_noavx512";
#else
  name += "_avx512";
#endif

  return name + ".csv";
}

/**
 * @brief Get the speedup file name based on compile-time configuration
 *
 * @return std::string File path for speedup results
 */
std::string getSpeedupFileName() {
  std::string name = "results/speedup/speedup";

#if PARALLEL == 0
  name += "_noparallel";
#else
  name += "_parallel";
#endif

#if USE_AVX512 == 0
  name += "_noavx512";
#else
  name += "_avx512";
#endif

  return name + ".csv";
}

/**
 * @brief Block size for chunking computations
 *
 * Defines the maximum number of floats to process in a single block.
 * Set to approximately 8192 floats (32KB) to:
 */
constexpr size_t BLOCK_SIZE =
    32 * 1024 / sizeof(float); // Approximately 8192 floats.

/**
 * @class AlignedAllocatorC17
 * @brief Memory allocator ensuring proper alignment for AVX operations
 *
 * C++17 implementation of an allocator that guarantees memory is aligned to 32
 * bytes, which is required for efficient AVX operations (256-bit registers).
 *
 * Uses the aligned new/delete operators introduced in C++17 to avoid manual
 * alignment calculations and prevent potential undefined behavior from
 * misaligned memory access.
 *
 * @tparam T Type of elements to allocate
 */
template <typename T> class AlignedAllocatorC17 {
public:
  using value_type = T;
  static constexpr size_t alignment = 32; // Alignment required for AVX

  /**
   * @brief Allocate aligned memory using C++17 aligned new operator
   *
   * @param n Number of elements to allocate
   * @return T* Pointer to aligned memory block
   */
  T *allocate(std::size_t n) {
    if (n == 0)
      return nullptr;
    return static_cast<T *>(
        ::operator new(n * sizeof(T), std::align_val_t(alignment)));
  }

  /**
   * @brief Deallocate memory using C++17 aligned delete operator
   *
   * @param p Pointer to memory block
   * @param Unused size parameter (required by allocator concept)
   */
  void deallocate(T *p, std::size_t) noexcept {
    ::operator delete(p, std::align_val_t(alignment));
  }

  /**
   * @brief Equality comparison operator (required by allocator concept)
   *
   * @tparam U Type parameter for the other allocator
   * @return true All allocators of this type compare equal
   */
  template <typename U>
  bool operator==(const AlignedAllocatorC17<U> &) const noexcept {
    return true;
  }

  /**
   * @brief Inequality comparison operator (required by allocator concept)
   *
   * @tparam U Type parameter for the other allocator
   * @return false All allocators of this type compare equal
   */
  template <typename U>
  bool operator!=(const AlignedAllocatorC17<U> &) const noexcept {
    return false;
  }
};

/**
 * @typedef aligned_vector
 * @brief Vector with guaranteed memory alignment for AVX operations
 *
 * We use this throughout the test framework to ensure that all data buffers
 * are properly aligned
 */
template <typename T>
using aligned_vector = std::vector<T, AlignedAllocatorC17<T>>;

void softmax_plain(const float *input, float *output, size_t K);
void softmax_auto(const float *input, float *output, size_t K,
                  int num_threads = -1);
void softmax_avx(const float *input, float *output, size_t K,
                 int num_threads = -1);
void softmax_avx_small(const float *input, float *output, size_t K,
                       int num_threads = -1);

/**
 * @brief Generate random input data with a fixed seed
 *
 * Creates deterministic random input data for consistent benchmarking.
 * Uses a fixed seed (42) to ensure reproducibility across test runs.
 *
 * @param K Number of elements to generate
 * @param min Minimum value (default: -1.0f)
 * @param max Maximum value (default: 1.0f)
 * @return aligned_vector<float> Vector of random floats
 */
aligned_vector<float> generate_random_input(size_t K, float min = -1.0f,
                                            float max = 1.0f) noexcept {
  aligned_vector<float> input(K);
  std::mt19937 gen(42); // Fixed seed for reproducibility
  std::uniform_real_distribution<float> dis(min, max);
  std::generate(input.begin(), input.end(), [&]() { return dis(gen); });
  return input;
}

/**
 * @brief Generate input data with controlled magnitude for stability testing
 *
 * Creates challenging test inputs with large dynamic range to stress numerical
 * stability. The magnitude scales logarithmically with input size to create
 * increasingly demanding test cases for larger arrays.
 *
 * @param K Number of elements to generate
 * @param scale_factor Controls the overall magnitude of values (default: 10.0f)
 * @return aligned_vector<float> Vector of values with wide magnitude range
 */
aligned_vector<float>
generate_stability_test_input(size_t K, float scale_factor = 10.0f) {
  aligned_vector<float> input(K);
  std::mt19937 gen(42); // Same seed for consistency

  // Scale magnitude based on K to stress test stability
  float magnitude = std::log10(static_cast<float>(K) + 1.0f) * scale_factor;
  std::uniform_real_distribution<float> dis(-magnitude, magnitude);

  std::generate(input.begin(), input.end(), [&]() { return dis(gen); });
  return input;
}

/**
 * @brief Verify that two arrays are approximately equal
 *
 * Compares two arrays using both absolute and relative error tolerances:
 * - Absolute error check: |a - b| <= abs_eps
 * - Relative error check: |a - b| <= rel_eps * max(|a|, |b|)
 *
 * This dual-threshold approach handles both small and large magnitudes well:
 * - Near zero, we rely on the absolute error threshold
 * - For larger values, we allow proportionally larger absolute differences
 *
 * @param a First array
 * @param b Second array
 * @param K Number of elements to compare
 * @param abs_eps Absolute error threshold (default: 1e-6)
 * @param rel_eps Relative error threshold (default: 1e-4)
 * @return bool True if arrays are approximately equal
 */
bool verify_results(const float *a, const float *b, size_t K,
                    float abs_eps = 1e-6, float rel_eps = 1e-4) noexcept {
  for (size_t i = 0; i < K; ++i) {
    float diff = std::abs(a[i] - b[i]);
    float max_val = std::max(std::abs(a[i]), std::abs(b[i]));
    if (diff > abs_eps && diff > rel_eps * max_val) {
      std::cerr << "Validation failed at index " << i << ":\n";
      std::cerr << "  Expected: " << a[i] << "\n";
      std::cerr << "  Got:      " << b[i] << "\n";
      std::cerr << "  Difference: " << diff << " (abs_eps: " << abs_eps
                << ", rel_eps*max: " << rel_eps * max_val << ")\n";
      return false;
    }
  }
  return true;
}

/**
 * @brief Validate that the output array is a proper softmax distribution
 *
 * Checks two essential properties of a softmax distribution:
 * 1. All values must be in the range [0,1]
 * 2. The sum of all values must be approximately 1.0
 *
 * Since floating-point arithmetic introduces rounding errors, we use an epsilon
 * threshold to allow for small deviations from the exact sum of 1.0.
 *
 * @param output Output array to validate
 * @param K Number of elements
 * @param epsilon Maximum allowed deviation from sum=1.0 (default: 1e-3)
 * @return bool True if the array represents a valid softmax distribution
 */
bool validate_softmax(const float *output, size_t K,
                      float epsilon = 1e-3) noexcept {
  float sum = 0.0f;
  for (size_t i = 0; i < K; ++i) {
    if (output[i] < 0 || output[i] > 1) {
      std::cerr << "Validation failed at index " << i
                << ": softmax output out of range [0,1].\n";
      std::cerr << "  Value: " << output[i] << "\n";
      return false;
    }
    sum += output[i];
  }
  if (std::abs(sum - 1.0f) > epsilon) {
    std::cerr << "Validation failed: softmax distribution sum is " << sum
              << " (expected 1.0) with tolerance epsilon = " << epsilon << "\n";
    return false;
  }
  return true;
}

/**
 * @brief Generic benchmark function for any softmax implementation
 *
 * Measures the execution time of a function using multiple samples and
 * iterations:
 * 1. Performs warmup runs to stabilize CPU frequency and caches
 * 2. Takes multiple timing samples to account for system variability
 * 3. Returns the median timing (more robust than mean against outliers)
 *
 * This is the most general form of the benchmark function that can handle
 * any number of additional arguments for the function being benchmarked.
 *
 * @tparam Func Type of the function to benchmark
 * @tparam Args Types of additional arguments to pass to the function
 * @param func Function to benchmark
 * @param input Input array
 * @param output Output array
 * @param K Number of elements
 * @param samples Number of timing samples to take (default: 15)
 * @param iterations_per_sample Number of iterations per sample (default: 30)
 * @param args Additional arguments to pass to the function
 * @return double Median execution time in seconds
 */
template <typename Func, typename... Args>
double benchmark(Func &&func, const float *input, float *output, size_t K,
                 size_t samples = 15, size_t iterations_per_sample = 30,
                 Args &&...args) noexcept {
  std::vector<double> measurements;
  measurements.reserve(samples);

  // Warmup phase to minimize startup overhead.
  for (size_t i = 0; i < 3; ++i) {
    std::forward<Func>(func)(input, output, K, std::forward<Args>(args)...);
  }

  // Measurement phase.
  for (size_t s = 0; s < samples; ++s) {
    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < iterations_per_sample; ++i) {
      std::forward<Func>(func)(input, output, K, std::forward<Args>(args)...);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    measurements.push_back(elapsed.count() / iterations_per_sample);
  }

  // Return the median of the measurements.
  std::sort(measurements.begin(), measurements.end());
  return measurements[measurements.size() / 2];
}

/**
 * @brief Specialized benchmark function for plain softmax implementation
 *
 * This variant handles the plain implementation which takes fewer arguments
 * (no thread count parameter). Using this specialized benchmark function
 * allows for cleaner code in the main benchmarking loop.
 *
 * @tparam Func Type of the function to benchmark
 * @param func Function to benchmark
 * @param input Input array
 * @param output Output array
 * @param K Number of elements
 * @param samples Number of timing samples to take (default: 15)
 * @param iterations_per_sample Number of iterations per sample (default: 30)
 * @return double Median execution time in seconds
 */
template <typename Func>
double benchmark_plain(Func &&func, const float *input, float *output, size_t K,
                       size_t samples = 15,
                       size_t iterations_per_sample = 30) noexcept {
  std::vector<double> measurements;
  measurements.reserve(samples);

  // Warmup phase
  for (size_t i = 0; i < 3; ++i) {
    std::forward<Func>(func)(input, output, K);
  }

  // Measurement phase
  for (size_t s = 0; s < samples; ++s) {
    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < iterations_per_sample; ++i) {
      std::forward<Func>(func)(input, output, K);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    measurements.push_back(elapsed.count() / iterations_per_sample);
  }

  std::sort(measurements.begin(), measurements.end());
  return measurements[measurements.size() / 2];
}

/**
 * @brief Specialized benchmark function for threaded softmax implementations
 *
 * This variant handles implementations that take a thread count parameter.
 * Structured similarly to benchmark_plain but passes the thread count to
 * the benchmarked function.
 *
 * @tparam Func Type of the function to benchmark
 * @param func Function to benchmark
 * @param input Input array
 * @param output Output array
 * @param K Number of elements
 * @param num_threads Number of threads to use (-1 for system default)
 * @param samples Number of timing samples to take (default: 15)
 * @param iterations_per_sample Number of iterations per sample (default: 30)
 * @return double Median execution time in seconds
 */
template <typename Func>
double benchmark_threaded(Func &&func, const float *input, float *output,
                          size_t K, int num_threads = -1, size_t samples = 15,
                          size_t iterations_per_sample = 30) noexcept {
  std::vector<double> measurements;
  measurements.reserve(samples);

  // Warmup phase
  for (size_t i = 0; i < 3; ++i) {
    std::forward<Func>(func)(input, output, K, num_threads);
  }

  // Measurement phase
  for (size_t s = 0; s < samples; ++s) {
    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < iterations_per_sample; ++i) {
      std::forward<Func>(func)(input, output, K, num_threads);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    measurements.push_back(elapsed.count() / iterations_per_sample);
  }

  std::sort(measurements.begin(), measurements.end());
  return measurements[measurements.size() / 2];
}

/**
 * @struct TestData
 * @brief Container for test data and results for a specific input size
 *
 * Groups all data related to one test case (input size K):
 * - The input array
 * - Output arrays for all three implementations
 *
 * This organization simplifies memory management and provides a clear
 * structure for the benchmark results.
 */
struct TestData {
  size_t size;                    // Number of elements.
  aligned_vector<float> input;    // Input array.
  aligned_vector<float> plain;    // Output from plain softmax.
  aligned_vector<float> auto_vec; // Output from auto-vectorized softmax.
  aligned_vector<float> avx;      // Output from AVX softmax.
};

// Forward declarations for specialized test functions
void test_numerical_stability();
void test_thread_scaling();

/**
 * @brief Main benchmark and testing harness for softmax implementations
 *
 * This function serves as the central testing framework that:
 * 1. Processes command-line arguments to determine test mode
 * 2. Generates a comprehensive range of test sizes (powers of 2 and random
 * values)
 * 3. Benchmarks three softmax implementations across all test sizes
 * 4. Records and analyzes performance data
 * 5. Categorizes results by array size for detailed performance insights
 *
 * The function offers several specialized test modes:
 * - Standard performance benchmarks (default)
 * - Thread scaling analysis (--thread-scaling)
 * - Numerical stability testing (--stability-test)
 * - Performance-only mode skipping stability tests (--performance-only)
 *
 * @param argc Number of command-line arguments
 * @param argv Array of command-line argument strings
 * @return int Exit status code (0 on success, non-zero on failure)
 */
int main(int argc, char *argv[]) {
  // Ensure the "results" folder exists.
  fs::create_directories("results");

  bool performance_only = false;

  // Check command line arguments
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--thread-scaling") {
      test_thread_scaling();
      return 0;
    } else if (std::string(argv[i]) == "--stability-test") {
      test_numerical_stability();
      return 0;
    } else if (std::string(argv[i]) == "--performance-only") {
      performance_only = true;
    }
  }

  int num_threads = -1; // Default: use system default

  std::vector<size_t> test_sizes;

  /**
   * Test size selection strategy:
   *
   * 1. Include all powers of 2 up to 2^22 (~128M elements)
   *    - Powers of 2 often reveal performance patterns related to:
   *      a) Cache line alignments
   *      b) Vector register width optimizations
   *      c) Memory allocation behaviors
   *
   * 2. Include 50 random non-power-of-2 sizes
   *    - Ensures testing with arbitrary array sizes
   *    - Helps detect performance issues specific to non-aligned sizes
   *    - Provides more realistic performance data across the size spectrum
   */
  std::set<size_t> unique_sizes;
  for (size_t power = 0; power <= 22; ++power) {
    size_t value = 1ULL << power; // 2^power
    test_sizes.push_back(value);
    unique_sizes.insert(value);
  }

  // Add 50 uniformly distributed values between 1 and 2^22
  std::mt19937 gen(
      42); // Using same seed as in generate_random_input for consistency
  std::uniform_int_distribution<size_t> dis(1, 1ULL << 22);

  size_t additional_needed = 50;
  while (additional_needed > 0) {
    size_t value = dis(gen);
    if (unique_sizes.find(value) == unique_sizes.end()) {
      test_sizes.push_back(value);
      unique_sizes.insert(value);
      additional_needed--;
    }
  }

  std::vector<size_t> sorted_test_sizes = test_sizes;
  std::sort(sorted_test_sizes.begin(), sorted_test_sizes.end());

  /**
   * Initialize test data for all input sizes
   *
   * Pre-allocating all test data upfront offers several advantages:
   * 1. Reduces memory allocation/deallocation overhead during benchmarking
   * 2. Ensures consistent memory layout throughout testing
   * 3. Separates data preparation from measurement for cleaner benchmarking
   */
  std::vector<TestData> test_data;
  for (auto K : sorted_test_sizes) {
    test_data.push_back({
        K,
        generate_random_input(K), // Random input with consistent seed
        aligned_vector<float>(K), // Plain softmax output buffer
        aligned_vector<float>(K), // Auto-vectorized softmax output buffer
        aligned_vector<float>(K)  // AVX softmax output buffer
    });
  }

  // Configure output file for raw benchmark results
  std::ofstream result_file(getResultFileName());
  if (!result_file) {
    std::cerr << "Failed to open " << getResultFileName() << "\n";
    return 1;
  }

  // Define variables to track benchmark results and performance order
  std::vector<BenchmarkResult>
      benchmark_results; // Yes, this is an array of structs, but we are not
                         // making any hpc computation here, just a few tests :)
  bool expected_order_maintained = true;
  std::vector<size_t> violated_sizes;

  // Write CSV header
  result_file << "Size,Plain,Auto,AVX\n";

  // Run benchmarks for each size
  std::cout << "Size       Plain       Auto        AVX\n";
  std::cout << "----------------------------------------\n";

  /**
   * Benchmark loop - core measurement process
   *
   * For each test size, we:
   * 1. Measure execution time for all three implementations
   * 2. Report immediate results to console for monitoring
   * 3. Record detailed results to CSV file for later analysis
   * 4. Check if the expected performance hierarchy is maintained
   */
  for (auto &data : test_data) {
    size_t K = data.size;

    // For plain version (baseline implementation)
    double t_plain =
        benchmark_plain(softmax_plain, data.input.data(), data.plain.data(), K);

    // For auto-vectorized version with num_threads
    double t_auto = benchmark_threaded(softmax_auto, data.input.data(),
                                       data.auto_vec.data(), K, num_threads);

    /**
     * Implementation selection strategy for AVX
     *
     * For smaller sizes (≤ 4*BLOCK_SIZE), use the specialized small
     * implementation:
     * - Reduces threading overhead for small arrays
     * - Optimized for lower memory footprint cases
     *
     * For larger sizes, use the standard AVX implementation:
     * - Better parallelism for large arrays
     * - More efficient memory access patterns for large datasets
     *
     * NOTE: I am using BLOCK_SIZE since I am working on a powerful server, for
     * a standard desktop CPU just BLOCK_SIZE should be enough.
     */
    double t_avx;
    if (K <= BLOCK_SIZE * 4)
      t_avx = benchmark_threaded(softmax_avx_small, data.input.data(),
                                 data.avx.data(), K, num_threads);
    else
      t_avx = benchmark_threaded(softmax_avx, data.input.data(),
                                 data.avx.data(), K, num_threads);

    // Save benchmark results for later analysis
    benchmark_results.push_back({K, t_plain, t_auto, t_avx});

    // Print benchmark results with formatted alignment for readability
    std::cout << std::left << std::setw(10) << K << std::fixed
              << std::setprecision(7) << std::setw(12) << t_plain
              << std::setw(12) << t_auto << std::setw(12) << t_avx << "\n";

    // Record to CSV for persistent storage
    result_file << K << "," << t_plain << "," << t_auto << "," << t_avx << "\n";
    if (!result_file) {
      std::cerr << "Failed to write results for size " << K << "\n";
      return 1;
    }

    /**
     * Performance order validation
     *
     * The expected performance hierarchy is:
     * Plain (slowest) ≥ Auto-vectorized ≥ AVX (fastest)
     */
    if (!(t_plain >= t_auto && t_auto >= t_avx)) {
      expected_order_maintained = false;
      violated_sizes.push_back(K);
    }
  }

  result_file.close();
  std::cout << "\nResults saved to " << getResultFileName() << "\n";

  /**
   * Speedup analysis - quantifying performance improvements
   *
   * This section:
   * 1. Calculates speedups relative to the scalar implementation
   * 2. Categorizes results by array size for targeted analysis
   * 3. Records speedup data to CSV for visualization
   */
  std::ofstream speedup_file(getSpeedupFileName());
  if (!speedup_file) {
    std::cerr << "Failed to open " << getSpeedupFileName() << "\n";
    return 1;
  }

  // Write header for speedup data
  speedup_file << "Size,Auto_Speedup,AVX_Speedup\n";

  // After running all benchmarks, analyze the results
  std::cout << "\nPerformance Analysis:\n";
  std::cout << "-------------------------------------------------\n";

  /**
   * Performance categorization
   *
   * Results are grouped into categories based on:
   * 1. Size range (tiny, small, medium, large)
   * 2. Whether the size is a power of 2
   *
   * This categorization helps identify:
   * - Which optimizations work best for which data sizes
   * - Whether certain size ranges benefit more from vectorization
   * - If power-of-2 sizes show different behavior than non-power-of-2
   */
  std::map<std::string, std::vector<double>> ratios;

  // Calculate speedups and categorize results
  for (const auto &result : benchmark_results) {
    size_t K = result.size;
    double auto_speedup = result.t_plain / result.t_auto;
    double avx_speedup = result.t_plain / result.t_avx;

    speedup_file << K << "," << auto_speedup << "," << avx_speedup << "\n";

    /**
     * Size categorization strategy:
     *
     * Four size categories:
     * - tiny:   K ≤ 64 elements (fits in few cache lines)
     * - small:  64 < K ≤ 1024 elements (fits in L1 cache)
     * - medium: 1024 < K ≤ 16384 elements (fits in L2 cache)
     * - large:  K > 16384 elements (exceeds typical L2 cache)
     */
    std::string category;
    if (K <= 64)
      category = "tiny";
    else if (K <= 1024)
      category = "small";
    else if (K <= 16384)
      category = "medium";
    else
      category = "large";

    // Also categorize by whether size is power of 2
    bool isPowerOf2 = (K & (K - 1)) == 0;
    std::string power_cat = isPowerOf2 ? "pow2" : "non-pow2";

    // Record speedups in appropriate categories
    ratios[category + "_auto"].push_back(auto_speedup);
    ratios[category + "_avx"].push_back(avx_speedup);
    ratios[power_cat + "_auto"].push_back(auto_speedup);
    ratios[power_cat + "_avx"].push_back(avx_speedup);

    // Further categorize non-power-of-2 sizes by small vs large
    if (!isPowerOf2) {
      if (K <= 16384) {
        ratios["small_non-pow2_auto"].push_back(auto_speedup);
        ratios["small_non-pow2_avx"].push_back(avx_speedup);
      } else {
        ratios["large_non-pow2_auto"].push_back(auto_speedup);
        ratios["large_non-pow2_avx"].push_back(avx_speedup);
      }
    }
  }

  speedup_file.close();
  std::cout << "Speedup results saved to " << getSpeedupFileName() << "\n";

  // Print size category definitions for reference
  std::cout << "- tiny:   K ≤ 64 elements\n";
  std::cout << "- small:  64 < K ≤ 1024 elements\n";
  std::cout << "- medium: 1024 < K ≤ 16384 elements\n";
  std::cout << "- large:  K > 16384 elements\n";
  std::cout << "- pow2:   Power-of-2 sizes\n";
  std::cout << "- non-pow2: Non-power-of-2 sizes\n";
  std::cout << "-------------------------------------------------\n";

  // Print average speedup for each category
  for (const auto &[category, values] : ratios) {
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    double avg = sum / values.size();
    std::cout << "Average speedup for " << category << ": " << std::fixed
              << std::setprecision(2) << avg << "x\n";
  }

  // Check if expected performance order was maintained
  if (!expected_order_maintained) {
    std::cout << "\nWARNING: Expected performance order (Plain ≥ Auto ≥ AVX) "
              << "was violated for " << violated_sizes.size()
              << " test sizes:\n";
    for (size_t i = 0; i < violated_sizes.size(); ++i) {
      std::cout << violated_sizes[i] << " ";
    }
    std::cout << std::endl;
  }

  /**
   * Run numerical stability test unless in performance-only mode
   *
   * Stability testing is important to verify that optimized implementations
   * maintain mathematical correctness, but can be skipped for quick performance
   * tests.
   */
  if (!performance_only) {
    test_numerical_stability();
  }

  return 0;
}

/**
 * @brief Tests numerical stability of different softmax implementations
 *
 * This function evaluates how well each softmax implementation maintains
 * the essential mathematical property that outputs sum to 1.0. It uses
 * inputs with high dynamic range to stress numerical precision.
 *
 * Test methodology:
 * 1. Generate inputs with large magnitude variations to stress floating-point
 * precision
 * 2. Run each implementation (plain, auto-vectorized, and AVX)
 * 3. Compute output sums using high precision (long double)
 * 4. Record deviations from the ideal sum of 1.0
 */
void test_numerical_stability() {
  // Create results directory if it doesn't exist
  fs::create_directories("results/numerical_stability");

  // Generate test sizes: powers of 2 plus 100 randomly distributed values
  std::vector<size_t> test_sizes;
  std::mt19937_64 gen(42);

  // Define size range from 1 to 2^30
  const size_t min_size = 1;
  const size_t max_size = 1ULL << 30;

  // Generate randomly distributed test sizes
  std::uniform_int_distribution<size_t> dis(min_size, max_size);
  std::set<size_t> unique_sizes;
  while (unique_sizes.size() < 50) {
    unique_sizes.insert(dis(gen));
  }

  // Include powers of 2 for detecting patterns related to vector register sizes
  for (size_t power = 0; power <= 30; ++power) {
    unique_sizes.insert(1ULL << power);
  }

  // Transfer to vector and sort for consistent output
  test_sizes.assign(unique_sizes.begin(), unique_sizes.end());
  std::sort(test_sizes.begin(), test_sizes.end());

  // Configure output filename based on build settings
  std::string filename = "results/numerical_stability/stability";

#if PARALLEL == 0
  filename += "_noparallel";
#else
  filename += "_parallel";
#endif

#if USE_AVX512 == 0
  filename += "_noavx512";
#else
  filename += "_avx512";
#endif

  filename += ".csv";

  std::ofstream stability_file(filename);
  if (!stability_file) {
    std::cerr << "Failed to open " << filename << " for writing\n";
    return;
  }

  // CSV header for sum values
  stability_file << "Size,PlainSum,AutoSum,AVXSum\n";

  std::cout << "\nTesting numerical stability...\n";
  std::cout << "Size      Plain Sum       Auto Sum        AVX Sum\n";
  std::cout << "------------------------------------------------\n";

  // Test each size
  for (auto K : test_sizes) {
    // Generate challenging input with large dynamic range to stress numerical
    // stability Scale factor 100.0f creates values large enough to cause
    // potential overflow/underflow
    aligned_vector<float> input = generate_stability_test_input(K, 100.0f);

    // Output buffers for each implementation
    aligned_vector<float> plain_output(K);
    aligned_vector<float> auto_output(K);
    aligned_vector<float> avx_output(K);

    // Execute all implementations
    softmax_plain(input.data(), plain_output.data(), K);
    softmax_auto(input.data(), auto_output.data(), K);

    // Use small optimized implementation for smaller sizes
    if (K <= BLOCK_SIZE * 4)
      softmax_avx_small(input.data(), avx_output.data(), K);
    else
      softmax_avx(input.data(), avx_output.data(), K);

    long double plain_sum = 0.0L;
    long double auto_sum = 0.0L;
    long double avx_sum = 0.0L;

    for (size_t i = 0; i < K; ++i) {
      plain_sum += static_cast<long double>(plain_output[i]);
      auto_sum += static_cast<long double>(auto_output[i]);
      avx_sum += static_cast<long double>(avx_output[i]);
    }

    // Print results with fixed precision formatting
    std::cout << std::left << std::setw(10) << K << std::fixed
              << std::setprecision(10) << std::setw(16) << plain_sum
              << std::setw(16) << auto_sum << std::setw(16) << avx_sum << "\n";

    // Write to CSV with extended precision (16 digits)
    stability_file << K << "," << std::setprecision(16) << plain_sum << ","
                   << std::setprecision(16) << auto_sum << ","
                   << std::setprecision(16) << avx_sum << "\n";
  }

  stability_file.close();
  std::cout << "\nNumerical stability results saved to " << filename << "\n";
}

/**
 * @brief Evaluates thread scaling efficiency for softmax implementations
 *
 * This function performs a thread scaling analysis by measuring how the
 * performance of auto-vectorized and AVX softmax implementations scales with an
 * increasing number of threads. The test uses a fixed large array size (8M
 * elements) and benchmarks each implementation with thread counts from 1 to the
 * maximum available threads on the system.
 *
 * Results are both displayed to the console and saved to a CSV file for further
 * analysis. The CSV output includes execution times for each implementation at
 * different thread counts.
 *
 * The function helps identify:
 * - Parallel scaling efficiency of each implementation
 * - Potential thread count saturation points
 * - Relative performance differences between implementations across thread
 * counts
 */
void test_thread_scaling() {
  // Fixed large size for meaningful thread scaling measurement
  const size_t K = 1ULL << 23; // ~8M elements

  // Create output directory structure
  fs::create_directories("results/thread_scaling");

  // Configure output filename based on build settings
  std::string filename = "results/thread_scaling/thread_scaling";
#if USE_AVX512 == 0
  filename += "_noavx512";
#else
  filename += "_avx512";
#endif
  filename += ".csv";

  // Open output file for results
  std::ofstream scaling_file(filename);
  if (!scaling_file) {
    std::cerr << "Failed to open " << filename << " for writing\n";
    return;
  }

  // Define CSV format
  scaling_file << "Threads,Auto,AVX\n";

  // Display benchmark header
  std::cout << "\nTesting thread scaling with size K=" << K << "...\n";
  std::cout << "Threads   Auto (s)      AVX (s)\n";
  std::cout << "--------------------------------\n";

  // Allocate memory for benchmark data (done once outside the loop)
  aligned_vector<float> input = generate_random_input(K);
  aligned_vector<float> auto_output(K);
  aligned_vector<float> avx_output(K);

  // Determine available parallelism on this system
  int max_threads = omp_get_max_threads();

  // Benchmark with increasing thread counts to measure scaling
  for (int num_threads = 1; num_threads <= max_threads; ++num_threads) {
    // Measure auto-vectorized implementation with specified thread count
    double t_auto = benchmark_threaded(softmax_auto, input.data(),
                                       auto_output.data(), K, num_threads);

    // Measure AVX implementation with specified thread count
    double t_avx = benchmark_threaded(softmax_avx, input.data(),
                                      avx_output.data(), K, num_threads);

    // Output results to console with formatted alignment
    std::cout << std::left << std::setw(10) << num_threads << std::fixed
              << std::setprecision(7) << std::setw(14) << t_auto
              << std::setw(12) << t_avx << "\n";

    // Record results to CSV file
    scaling_file << num_threads << "," << t_auto << "," << t_avx << "\n";
  }

  scaling_file.close();
  std::cout << "\nThread scaling results saved to " << filename << "\n";
}
