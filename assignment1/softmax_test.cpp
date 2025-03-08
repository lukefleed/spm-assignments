// File: softmax_test.cpp
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream> // Add this include for file operations
#include <hpc_helpers.hpp>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

// Dichiarazioni delle funzioni di softmax
void softmax_plain(const float *input, float *output, size_t K);
void softmax_auto(const float *input, float *output, size_t K);
void softmax_avx(const float *input, float *output, size_t K);

// Genera input condiviso per tutti i test
std::vector<float> generate_random_input(size_t K, float min = -1.0f,
                                         float max = 1.0f) {
  std::vector<float> input(K);
  // std::random_device rd;
  // std::mt19937 gen(rd());
  std::mt19937 gen(5489); // fixed seed for reproducible results
  std::uniform_real_distribution<float> dis(min, max);
  for (size_t i = 0; i < K; ++i) {
    input[i] = dis(gen);
  }
  return input;
}

// Verifica la correttezza degli output
bool verify_results(const float *a, const float *b, size_t K,
                    float epsilon = 1e-6) {
  for (size_t i = 0; i < K; ++i) {
    if (std::abs(a[i] - b[i]) > epsilon) {
      std::cerr << "Mismatch at position " << i << ": " << a[i] << " vs "
                << b[i] << std::endl;
      return false;
    }
  }
  return true;
}

// Template per il test delle prestazioni con statistiche migliorate
template <typename Func>
double benchmark(Func &&func, const float *input, float *output, size_t K,
                 size_t samples = 10, size_t iterations_per_sample = 5) {
  std::vector<double> measurements;
  measurements.reserve(samples);

  // Esegue multiple misurazioni indipendenti
  for (size_t s = 0; s < samples; ++s) {
    auto start = std::chrono::high_resolution_clock::now();

    // Ogni misurazione esegue la funzione più volte
    for (size_t i = 0; i < iterations_per_sample; ++i) {
      func(input, output, K);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double avg_time = elapsed.count() / iterations_per_sample;
    measurements.push_back(avg_time);
  }

  // Calcola la media delle misurazioni
  double sum = 0.0;
  for (double time : measurements) {
    sum += time;
  }

  return sum / samples;
}

int main() {
  // Dimensioni da testare
  const std::vector<size_t> test_sizes = {
      128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072};

  // Intestazione della tabella
  std::printf("%-10s %-10s %-10s %-10s\n", "Size", "Plain", "Auto", "AVX");
  std::printf("----------------------------------------\n");

  // Create and open output CSV file
  std::ofstream result_file("results.csv");
  if (!result_file.is_open()) {
    std::cerr << "Failed to open results.csv for writing" << std::endl;
    return 1;
  }

  // Write CSV header
  result_file << "Size,Plain,Auto,AVX" << std::endl;

  bool expected_order_maintained = true;
  std::vector<size_t> violated_sizes;

  for (size_t K : test_sizes) {
    // Genera input condiviso
    auto input = generate_random_input(K);
    std::vector<float> output_plain(K);
    std::vector<float> output_auto(K);
    std::vector<float> output_avx(K);

    // Warm-up della cache
    softmax_plain(input.data(), output_plain.data(), K);
    softmax_auto(input.data(), output_auto.data(), K);
    softmax_avx(input.data(), output_avx.data(), K);

    // Verifica correttezza
    if (!verify_results(output_plain.data(), output_auto.data(), K) ||
        !verify_results(output_plain.data(), output_avx.data(), K)) {
      std::cerr << "Validation failed for size " << K << std::endl;
      return 1;
    }

    // Benchmark delle prestazioni
    const double t_plain =
        benchmark(softmax_plain, input.data(), output_plain.data(), K);
    const double t_auto =
        benchmark(softmax_auto, input.data(), output_auto.data(), K);
    const double t_avx =
        benchmark(softmax_avx, input.data(), output_avx.data(), K);

    // Stampa risultati
    std::printf("%-10zu %-10.7f %-10.7f %-10.7f\n", K, t_plain, t_auto, t_avx);

    // Write results to CSV file
    result_file << K << "," << t_plain << "," << t_auto << "," << t_avx
                << std::endl;

    // Verifica ordine di efficienza atteso
    if (!(t_plain >= t_auto && t_auto >= t_avx)) {
      expected_order_maintained = false;
      violated_sizes.push_back(K);
    }
  }

  std::printf("----------------------------------------\n");
  if (!expected_order_maintained) {
    std::printf("ATTENZIONE: L'ordine di efficienza atteso (Plain > Auto > "
                "AVX) non è stato rispettato\n");
    std::printf("Dimensioni problematiche: ");
    for (size_t i = 0; i < violated_sizes.size(); ++i) {
      std::printf("%zu", violated_sizes[i]);
      if (i < violated_sizes.size() - 1)
        std::printf(", ");
    }
    std::printf("\n");

    result_file.close();
    std::cout << "Results saved to results.csv" << std::endl;

    return 0;
  }
}
