#include "theoretical_analysis.h"
#include "collatz.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

/**
 * @brief Analyzes the theoretical metrics of a given workload.
 *
 * Calculates total work, critical path, and parallelism for a set of ranges.
 *
 * @param ranges A vector of Range objects representing the workload.
 * @return TheoreticalMetrics struct containing the analysis results.
 */
TheoreticalMetrics analyze_workload_theory(const std::vector<Range> &ranges) {
  TheoreticalMetrics metrics;
  metrics.total_work = 0;
  metrics.critical_path = 0;

  // Analyze each range in the workload
  for (const auto &range : ranges) {
    if (range.start > range.end)
      continue;

    ull start = (range.start == 0) ? 1 : range.start;

    for (ull n = start; n <= range.end; ++n) {
      try {
        ull steps = collatz_steps(n);
        metrics.total_work += steps;
        metrics.critical_path = std::max(metrics.critical_path, steps);

        // Handle loop termination for max value to prevent infinite loop
        if (n == std::numeric_limits<ull>::max())
          break;
      } catch (const std::overflow_error &e) {
        std::cerr << "Warning: Overflow in theoretical analysis for n=" << n
                  << std::endl;
        break;
      }
    }
  }

  // Calculate parallelism (W/S)
  metrics.parallelism =
      (metrics.critical_path > 0)
          ? static_cast<double>(metrics.total_work) / metrics.critical_path
          : 0.0;

  return metrics;
}

/**
 * @brief Generates a CSV file containing theoretical speedup data for different
 * workloads.
 *
 * Analyzes each workload and writes the results (WorkloadID,
 * WorkloadDescription, WorkTotal, CriticalPath, Parallelism) to a CSV file.
 *
 * @param workloads A vector of vectors of Range objects, representing multiple
 * workloads.
 * @param workload_descriptions A vector of strings, providing descriptions for
 * each workload.
 * @param output_filename The name of the CSV file to be generated.
 * @return True if the CSV file was successfully generated, false otherwise.
 */
bool generate_theoretical_speedup_csv(
    const std::vector<std::vector<Range>> &workloads,
    const std::vector<std::string> &workload_descriptions,
    const std::string &output_filename) {

  // Open output file
  std::ofstream csv_file(output_filename);
  if (!csv_file.is_open()) {
    std::cerr << "Error: Could not open file " << output_filename
              << " for writing." << std::endl;
    return false;
  }

  // Write CSV header
  csv_file
      << "WorkloadID,WorkloadDescription,WorkTotal,CriticalPath,Parallelism"
      << std::endl;

  // Analyze each workload
  for (size_t workload_idx = 0; workload_idx < workloads.size();
       ++workload_idx) {
    const auto &workload = workloads[workload_idx];
    const auto &description = workload_descriptions[workload_idx];

    std::cout << "Analyzing workload " << workload_idx << ": " << description
              << "..." << std::endl;

    // Calculate theoretical metrics
    TheoreticalMetrics metrics = analyze_workload_theory(workload);

    std::cout << "  Work (W): " << metrics.total_work << std::endl;
    std::cout << "  Span (S): " << metrics.critical_path << std::endl;
    std::cout << "  Parallelism (W/S): " << metrics.parallelism << std::endl;

    // Write workload data to CSV file
    csv_file << workload_idx << ","
             << "\"" << description << "\"," << metrics.total_work << ","
             << metrics.critical_path << "," << std::fixed
             << std::setprecision(4) << metrics.parallelism << std::endl;
  }

  csv_file.close();
  return true;
}
