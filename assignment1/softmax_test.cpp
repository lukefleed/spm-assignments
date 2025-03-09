// File: softmax_test.cpp
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib> // per posix_memalign
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

/**
 * @brief Aligned allocator for proper memory alignment with SIMD instructions
 * @tparam T The type of elements to allocate
 */
template <typename T> class AlignedAllocator {
public:
  using value_type = T;
  static constexpr size_t alignment = 32; // Alignment at 32 bytes for AVX

  // Verify alignment is a power of 2 and sufficient for AVX
  static_assert((alignment & (alignment - 1)) == 0,
                "Alignment must be a power of 2");
  static_assert(alignment >= 32,
                "Alignment must be at least 32 bytes for AVX operations");

  AlignedAllocator() noexcept = default;

  template <typename U>
  AlignedAllocator(const AlignedAllocator<U> &) noexcept {}

  /**
   * @brief Allocate aligned memory
   * @param n Number of elements to allocate
   * @return Pointer to aligned memory
   * @throws std::bad_alloc if allocation fails
   */
  T *allocate(size_t n) {
    if (n == 0)
      return nullptr;

    void *ptr = nullptr;
    if (posix_memalign(&ptr, alignment, n * sizeof(T)) != 0) {
      throw std::bad_alloc();
    }
    return static_cast<T *>(ptr);
  }

  /**
   * @brief Deallocate previously allocated memory
   * @param p Pointer to memory to deallocate
   * @param n Number of elements (unused)
   */
  void deallocate(T *p, size_t) noexcept { free(p); }

  template <typename U>
  bool operator==(const AlignedAllocator<U> &) const noexcept {
    return true;
  }

  template <typename U>
  bool operator!=(const AlignedAllocator<U> &) const noexcept {
    return false;
  }
};

template <typename T>
using aligned_vector = std::vector<T, AlignedAllocator<T>>;

// Dichiarazioni delle funzioni softmax
void softmax_plain(const float *input, float *output, size_t K);
void softmax_auto(const float *input, float *output, size_t K);
void softmax_avx(const float *input, float *output, size_t K);
void softmax_avx_small(const float *input, float *output, size_t K);

/**
 * @brief Generate random input data for testing with a fixed seed
 * @param K Size of the input array
 * @param min Minimum value for random numbers
 * @param max Maximum value for random numbers
 * @return Vector of random floats aligned for SIMD operations
 */
aligned_vector<float> generate_random_input(size_t K, float min = -1.0f,
                                            float max = 1.0f) noexcept {
  aligned_vector<float> input(K);
  std::mt19937 gen(5489); // Seed fisso per risultati riproducibili
  std::uniform_real_distribution<float> dis(min, max);
  std::generate(input.begin(), input.end(), [&]() { return dis(gen); });
  return input;
}

/**
 * @brief Verify if two result arrays are approximately equal
 * @param a First array to compare
 * @param b Second array to compare
 * @param K Size of the arrays
 * @param abs_eps Absolute error tolerance
 * @param rel_eps Relative error tolerance
 * @return True if results match within tolerance, false otherwise
 */
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

/**
 * @brief Validate that the output array satisfies softmax properties
 * @param output Array containing softmax results
 * @param K Size of the array
 * @param epsilon Error tolerance for the sum (should be close to 1.0)
 * @return True if output is a valid softmax result, false otherwise
 */
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

/**
 * @brief Benchmark a function with warmup and return the median execution time
 * @tparam Func Type of the function to benchmark
 * @param func Function to benchmark
 * @param input Input data array
 * @param output Output data array
 * @param K Size of the arrays
 * @param samples Number of measurement samples to collect
 * @param iterations_per_sample Number of iterations per sample
 * @return Median execution time in seconds
 */
template <typename Func>
double benchmark(Func &&func, const float *input, float *output, size_t K,
                 size_t samples = 11,
                 size_t iterations_per_sample = 20) noexcept {
  std::vector<double> measurements;
  measurements.reserve(samples);

  // Warmup
  for (size_t i = 0; i < 3; ++i) {
    func(input, output, K);
  }

  // Misurazioni
  for (size_t s = 0; s < samples; ++s) {
    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < iterations_per_sample; ++i) {
      func(input, output, K);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    measurements.push_back(elapsed.count() / iterations_per_sample);
  }

  // Restituisce la mediana
  std::sort(measurements.begin(), measurements.end());
  return measurements[measurements.size() / 2];
}

/**
 * @brief Structure to hold test data for each test size
 */
struct TestData {
  size_t size;
  aligned_vector<float> input;
  aligned_vector<float> plain;
  aligned_vector<float> auto_vec;
  aligned_vector<float> avx;
};

/**
 * @brief Main function to run the softmax implementation benchmarks
 * @return 0 on success, 1 on failure
 */
int main() {
  const std::vector<size_t> test_sizes = {128,    256,    512,    1024,   2048,
                                          4096,   8192,   16384,  32768,  65536,
                                          131072, 262144, 524288, 1048576};

  // Initialize test data
  std::vector<TestData> test_data;
  for (auto K : test_sizes) {
    test_data.push_back({K, generate_random_input(K), aligned_vector<float>(K),
                         aligned_vector<float>(K), aligned_vector<float>(K)});
  }

  // Open the result file
  std::ofstream result_file("results.csv");
  if (!result_file) {
    std::cerr << "Failed to open results.csv\n";
    return 1;
  }

  // Print header
  std::cout << std::left << std::setw(10) << "Size" << std::setw(12) << "Plain"
            << std::setw(12) << "Auto" << std::setw(12) << "AVX"
            << "\n-------------------------------------------------\n";

  result_file << "Size,Plain,Auto,AVX\n";

  bool expected_order_maintained = true;
  std::vector<size_t> violated_sizes;

  for (auto &data : test_data) {
    const size_t K = data.size;

    // Cache warmup
    softmax_plain(data.input.data(), data.plain.data(), K);
    softmax_auto(data.input.data(), data.auto_vec.data(), K);

    // Use softmax_avx_small for smaller sizes
    if (K <= 8192) {
      softmax_avx_small(data.input.data(), data.avx.data(), K);
    } else {
      softmax_avx(data.input.data(), data.avx.data(), K);
    }

    // Validate results
    if (!verify_results(data.plain.data(), data.auto_vec.data(), K) ||
        !verify_results(data.plain.data(), data.avx.data(), K) ||
        !validate_softmax(data.plain.data(), K) ||
        !validate_softmax(data.auto_vec.data(), K) ||
        !validate_softmax(data.avx.data(), K)) {
      std::cerr << "Validation failed for size " << K << "\n";
      return 1;
    }

    // Benchmark
    const double t_plain =
        benchmark(softmax_plain, data.input.data(), data.plain.data(), K);
    const double t_auto =
        benchmark(softmax_auto, data.input.data(), data.auto_vec.data(), K);

    // Use appropriate function for benchmark based on size
    double t_avx;
    if (K <= 8192) {
      t_avx =
          benchmark(softmax_avx_small, data.input.data(), data.avx.data(), K);
    } else {
      t_avx = benchmark(softmax_avx, data.input.data(), data.avx.data(), K);
    }

    // Print results
    std::cout << std::left << std::setw(10) << K << std::fixed
              << std::setprecision(7) << std::setw(12) << t_plain
              << std::setw(12) << t_auto << std::setw(12) << t_avx << "\n";

    // Write results to file
    result_file << K << "," << t_plain << "," << t_auto << "," << t_avx << "\n";
    if (!result_file) {
      std::cerr << "Failed to write results for size " << K << "\n";
      return 1;
    }

    // Check if the expected order is maintained
    if (!(t_plain >= t_auto && t_auto >= t_avx)) {
      expected_order_maintained = false;
      violated_sizes.push_back(K);
    }
  }

  // Print summary
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
  return 0;
}
