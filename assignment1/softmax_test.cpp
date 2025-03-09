// File: softmax_test.cpp
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map> // For std::map
#include <memory>
#include <new>     // For std::align_val_t
#include <numeric> // For std::accumulate
#include <random>
#include <vector>

constexpr size_t BLOCK_SIZE =
    32 * 1024 / sizeof(float); // Approximately 8192 floats.

// This allocator uses the over-aligned new and delete operators provided in
// C++17 to guarantee that allocated memory is aligned to 32 bytes (suitable for
// AVX).
template <typename T> class AlignedAllocatorC17 {
public:
  using value_type = T;
  static constexpr size_t alignment = 32; // Alignment required for AVX

  // Allocate memory using the C++17 aligned operator new.
  T *allocate(std::size_t n) {
    if (n == 0)
      return nullptr;
    return static_cast<T *>(
        ::operator new(n * sizeof(T), std::align_val_t(alignment)));
  }

  // Deallocate memory using the C++17 aligned operator delete.
  void deallocate(T *p, std::size_t) noexcept {
    ::operator delete(p, std::align_val_t(alignment));
  }

  template <typename U>
  bool operator==(const AlignedAllocatorC17<U> &) const noexcept {
    return true;
  }

  template <typename U>
  bool operator!=(const AlignedAllocatorC17<U> &) const noexcept {
    return false;
  }
};

// Alias for a vector using the C++17 aligned allocator.
template <typename T>
using aligned_vector = std::vector<T, AlignedAllocatorC17<T>>;

//------------------------------------------------------------------------------
// Declarations for softmax functions to be tested.
// Implementations should compute:
//   output[i] = exp(input[i] - max(input)) / sum_j(exp(input[j] - max(input)))
//------------------------------------------------------------------------------
void softmax_plain(const float *input, float *output, size_t K);
void softmax_auto(const float *input, float *output, size_t K);
void softmax_avx(const float *input, float *output, size_t K);
void softmax_avx_small(const float *input, float *output, size_t K);

//------------------------------------------------------------------------------
// Generate random input data with a fixed seed
//------------------------------------------------------------------------------
aligned_vector<float> generate_random_input(size_t K, float min = -1.0f,
                                            float max = 1.0f) noexcept {
  aligned_vector<float> input(K);
  std::mt19937 gen(5489); // Fixed seed for reproducible results.
  std::uniform_real_distribution<float> dis(min, max);
  std::generate(input.begin(), input.end(), [&]() { return dis(gen); });
  return input;
}

//------------------------------------------------------------------------------
// Verify that two arrays are approximately equal.
// Uses both absolute and relative error tolerances.
//------------------------------------------------------------------------------
bool verify_results(const float *a, const float *b, size_t K,
                    float abs_eps = 1e-6, float rel_eps = 1e-4) noexcept {
  for (size_t i = 0; i < K; ++i) {
    float diff = std::abs(a[i] - b[i]);
    float max_val = std::max(std::abs(a[i]), std::abs(b[i]));
    if (diff > abs_eps && diff > rel_eps * max_val) {
      std::cerr << "Mismatch at " << i << ": " << a[i] << " vs " << b[i]
                << "\n";
      return false;
    }
  }
  return true;
}

//------------------------------------------------------------------------------
// Validate that the output array is a proper softmax distribution:
// all values are in [0,1] and sum approximately to 1.
//------------------------------------------------------------------------------
bool validate_softmax(const float *output, size_t K,
                      float epsilon = 1e-3) noexcept {
  float sum = 0.0f;
  for (size_t i = 0; i < K; ++i) {
    if (output[i] < 0 || output[i] > 1)
      return false;
    sum += output[i];
  }
  return std::abs(sum - 1.0f) <= epsilon;
}

//------------------------------------------------------------------------------
// Benchmark a softmax function by running multiple iterations and samples,
// then return the median execution time (seconds).
//------------------------------------------------------------------------------
template <typename Func>
double benchmark(Func &&func, const float *input, float *output, size_t K,
                 size_t samples = 11,
                 size_t iterations_per_sample = 20) noexcept {
  std::vector<double> measurements;
  measurements.reserve(samples);

  // Warmup phase to minimize startup overhead.
  for (size_t i = 0; i < 3; ++i) {
    func(input, output, K);
  }

  // Measurement phase.
  for (size_t s = 0; s < samples; ++s) {
    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < iterations_per_sample; ++i) {
      func(input, output, K);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    measurements.push_back(elapsed.count() / iterations_per_sample);
  }

  // Return the median of the measurements.
  std::sort(measurements.begin(), measurements.end());
  return measurements[measurements.size() / 2];
}

//------------------------------------------------------------------------------
// Structure to hold test data for each test size.
//------------------------------------------------------------------------------
struct TestData {
  size_t size;                    // Number of elements.
  aligned_vector<float> input;    // Input array.
  aligned_vector<float> plain;    // Output from plain softmax.
  aligned_vector<float> auto_vec; // Output from auto-vectorized softmax.
  aligned_vector<float> avx;      // Output from AVX softmax.
};

//------------------------------------------------------------------------------
// Main function: runs tests, benchmarks, and writes results to CSV.
//------------------------------------------------------------------------------
int main() {
  // Define various input sizes for testing.
  const std::vector<size_t> test_sizes = {
      // Small sizes to catch edge cases
      1, 2, 3, 4, 8, 16, 32, 63, 64, 65, 127,

      // Standard powers of 2
      128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072,
      262144, 524288, 1048576,

      // Non-power-of-2 sizes to test alignment handling
      100, 500, 1000, 3000, 7000, 10000, 50000, 100000, 500000,

      // Sizes near multiples of cache line (64 bytes = 16 floats)
      112, 120, 136, 144, 240, 272,

      // Sizes near AVX register boundaries (8 floats)
      7, 9, 15, 17, 23, 25, 31, 33};

  // Sort the test sizes for better output organization
  std::vector<size_t> sorted_test_sizes = test_sizes;
  std::sort(sorted_test_sizes.begin(), sorted_test_sizes.end());

  // Initialize test data for each input size.
  std::vector<TestData> test_data;
  for (auto K : sorted_test_sizes) {
    test_data.push_back({
        K,
        generate_random_input(K), // Random input.
        aligned_vector<float>(K), // Plain softmax output.
        aligned_vector<float>(K), // Auto-vectorized softmax output.
        aligned_vector<float>(K)  // AVX softmax output.
    });
  }

  // Open CSV file for writing benchmark results.
  std::ofstream result_file("results.csv");
  if (!result_file) {
    std::cerr << "Failed to open results.csv\n";
    return 1;
  }

  // Print header for console output.
  std::cout << std::left << std::setw(10) << "Size" << std::setw(12) << "Plain"
            << std::setw(12) << "Auto" << std::setw(12) << "AVX"
            << "\n"
            << "-------------------------------------------------\n";

  result_file << "Size,Plain,Auto,AVX\n";

  bool expected_order_maintained = true;
  std::vector<size_t> violated_sizes;

  for (auto &data : test_data) {
    const size_t K = data.size;

    // Warmup: run plain and auto implementations.
    softmax_plain(data.input.data(), data.plain.data(), K);
    softmax_auto(data.input.data(), data.auto_vec.data(), K);

    // Choose the appropriate AVX function based on input size.
    if (K <= 8192)
      softmax_avx_small(data.input.data(), data.avx.data(), K);
    else
      softmax_avx(data.input.data(), data.avx.data(), K);

    // Validate that all implementations yield equivalent results and valid
    // softmax outputs.
    if (!verify_results(data.plain.data(), data.auto_vec.data(), K) ||
        !verify_results(data.plain.data(), data.avx.data(), K) ||
        !validate_softmax(data.plain.data(), K) ||
        !validate_softmax(data.auto_vec.data(), K) ||
        !validate_softmax(data.avx.data(), K)) {
      std::cerr << "Validation failed for size " << K << "\n";
      return 1;
    }

    // Benchmark each implementation.
    const double t_plain =
        benchmark(softmax_plain, data.input.data(), data.plain.data(), K);
    const double t_auto =
        benchmark(softmax_auto, data.input.data(), data.auto_vec.data(), K);

    // Use a heuristic based on cache size to select the appropriate AVX
    // function.
    double t_avx;
    if (K <= BLOCK_SIZE * 2)
      t_avx =
          benchmark(softmax_avx_small, data.input.data(), data.avx.data(), K);
    else
      t_avx = benchmark(softmax_avx, data.input.data(), data.avx.data(), K);

    // Print benchmark results.
    std::cout << std::left << std::setw(10) << K << std::fixed
              << std::setprecision(7) << std::setw(12) << t_plain
              << std::setw(12) << t_auto << std::setw(12) << t_avx << "\n";

    result_file << K << "," << t_plain << "," << t_auto << "," << t_avx << "\n";
    if (!result_file) {
      std::cerr << "Failed to write results for size " << K << "\n";
      return 1;
    }

    // Check if the expected performance order (Plain ≥ Auto ≥ AVX) holds.
    if (!(t_plain >= t_auto && t_auto >= t_avx)) {
      expected_order_maintained = false;
      violated_sizes.push_back(K);
    }
  }

  // Print a summary of any performance order violations.
  std::cout << "-------------------------------------------------\n";
  if (!expected_order_maintained) {
    std::cout
        << "WARNING: Expected performance order (Plain ≥ Auto ≥ AVX) violated\n"
        << "Problematic sizes: ";
    for (size_t i = 0; i < violated_sizes.size(); ++i) {
      std::cout << violated_sizes[i];
      if (i < violated_sizes.size() - 1)
        std::cout << ", ";
    }
    std::cout << "\n";
  }

  result_file.close();
  std::cout << "\nResults saved to results.csv\n";

  // After running all benchmarks, analyze the results
  std::cout << "\nPerformance Analysis:\n";
  std::cout << "-------------------------------------------------\n";

  // Map to store performance ratios by size category
  std::map<std::string, std::vector<double>> ratios;

  for (auto &data : test_data) {
    const size_t K = data.size;
    double plain_time =
        benchmark(softmax_plain, data.input.data(), data.plain.data(), K);
    double auto_time =
        benchmark(softmax_auto, data.input.data(), data.auto_vec.data(), K);
    double avx_time =
        (K <= BLOCK_SIZE * 2)
            ? benchmark(softmax_avx_small, data.input.data(), data.avx.data(),
                        K)
            : benchmark(softmax_avx, data.input.data(), data.avx.data(), K);

    double auto_speedup = plain_time / auto_time;
    double avx_speedup = plain_time / avx_time;

    // Categorize the size
    std::string category;
    if (K <= 64)
      category = "tiny";
    else if (K <= 1024)
      category = "small";
    else if (K <= 16384)
      category = "medium";
    else
      category = "large";

    // Is it a power of 2?
    bool isPowerOf2 = (K & (K - 1)) == 0;
    std::string power_cat = isPowerOf2 ? "pow2" : "non-pow2";

    // Store the ratios
    ratios[category + "_auto"].push_back(auto_speedup);
    ratios[category + "_avx"].push_back(avx_speedup);
    ratios[power_cat + "_auto"].push_back(auto_speedup);
    ratios[power_cat + "_avx"].push_back(avx_speedup);

    // Divide non-power-of-2 into small and large categories
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

  std::cout << "- tiny:   K ≤ 64 elements\n";
  std::cout << "- small:  64 < K ≤ 1024 elements\n";
  std::cout << "- medium: 1024 < K ≤ 16384 elements\n";
  std::cout << "- large:  K > 16384 elements\n";
  std::cout << "- pow2:   Power-of-2 sizes\n";
  std::cout << "- non-pow2: Non-power-of-2 sizes\n";
  std::cout << "-------------------------------------------------\n";

  // Print average speedups by category
  for (const auto &[category, values] : ratios) {
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    double avg = sum / values.size();
    std::cout << "Average speedup for " << category << ": " << std::fixed
              << std::setprecision(2) << avg << "x\n";
  }

  return 0;
}
