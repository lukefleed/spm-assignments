#include "../common/record.hpp"
#include "../common/timer.hpp"
#include "../common/utils.hpp"
#include "../sequential/sequential_mergesort.hpp"
#include <iomanip>
#include <iostream>

// External functions
void ff_pipeline_mergesort(std::vector<std::unique_ptr<Record>> &data,
                           size_t nworkers);
void ff_farm_mergesort(std::vector<std::unique_ptr<Record>> &data,
                       size_t nworkers);

int main(int argc, char *argv[]) {
  Config config = parse_args(argc, argv);

  std::cout << "=== Single Node MergeSort Comparison ===\n";
  std::cout << "Array size: " << config.array_size << " elements\n";
  std::cout << "Payload size: " << config.payload_size << " bytes\n";
  std::cout << "Total data: "
            << format_bytes(config.array_size * (8 + config.payload_size))
            << "\n";
  std::cout << "Threads: " << config.num_threads << "\n";
  std::cout << "Pattern: ";
  switch (config.pattern) {
  case DataPattern::RANDOM:
    std::cout << "Random\n";
    break;
  case DataPattern::SORTED:
    std::cout << "Already Sorted\n";
    break;
  case DataPattern::REVERSE_SORTED:
    std::cout << "Reverse Sorted\n";
    break;
  case DataPattern::NEARLY_SORTED:
    std::cout << "Nearly Sorted\n";
    break;
  }
  std::cout << "\n";

  // Generate test data
  auto original_data =
      generate_data(config.array_size, config.payload_size, config.pattern);

  // Results table
  std::cout << std::setw(25) << "Implementation" << std::setw(15) << "Time (ms)"
            << std::setw(15) << "Speedup" << std::setw(15) << "Valid\n";
  std::cout << std::string(70, '-') << "\n";

  double baseline_time = 0;

  // Test std::sort
  {
    auto data = copy_records(original_data, config.payload_size);
    Timer t;
    stl_sort(data);
    double ms = t.elapsed_ms();
    baseline_time = ms;

    bool valid = config.validate ? is_sorted(data) : true;
    std::cout << std::setw(25) << "std::sort" << std::setw(15) << std::fixed
              << std::setprecision(2) << ms << std::setw(15) << "1.00x"
              << std::setw(15) << (valid ? "✓" : "✗") << "\n";
  }

  // Test sequential mergesort
  {
    auto data = copy_records(original_data, config.payload_size);
    Timer t;
    sequential_mergesort(data);
    double ms = t.elapsed_ms();

    bool valid = config.validate ? is_sorted(data) : true;
    std::cout << std::setw(25) << "Sequential MergeSort" << std::setw(15)
              << std::fixed << std::setprecision(2) << ms << std::setw(15)
              << std::fixed << std::setprecision(2) << baseline_time / ms << "x"
              << std::setw(15) << (valid ? "✓" : "✗") << "\n";
  }

  // Test FastFlow pipeline
  {
    auto data = copy_records(original_data, config.payload_size);
    Timer t;
    ff_pipeline_mergesort(data, config.num_threads);
    double ms = t.elapsed_ms();

    bool valid = config.validate ? is_sorted(data) : true;
    std::cout << std::setw(25) << "FF Pipeline+Farm" << std::setw(15)
              << std::fixed << std::setprecision(2) << ms << std::setw(15)
              << std::fixed << std::setprecision(2) << baseline_time / ms << "x"
              << std::setw(15) << (valid ? "✓" : "✗") << "\n";
  }

  // Test FastFlow farm with feedback
  if (config.array_size < 10000000) { // Limit for recursive approach
    auto data = copy_records(original_data, config.payload_size);
    Timer t;
    ff_farm_mergesort(data, config.num_threads);
    double ms = t.elapsed_ms();

    bool valid = config.validate ? is_sorted(data) : true;
    std::cout << std::setw(25) << "FF Farm+Feedback" << std::setw(15)
              << std::fixed << std::setprecision(2) << ms << std::setw(15)
              << std::fixed << std::setprecision(2) << baseline_time / ms << "x"
              << std::setw(15) << (valid ? "✓" : "✗") << "\n";
  }

  return 0;
}
