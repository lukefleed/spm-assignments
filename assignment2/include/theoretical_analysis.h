#ifndef THEORETICAL_ANALYSIS_H
#define THEORETICAL_ANALYSIS_H

#include "common_types.h"
#include <string>
#include <vector>

struct TheoreticalMetrics {
  ull total_work;     // Total operations (steps) across all numbers
  ull critical_path;  // Maximum steps of any single number
  double parallelism; // W/S ratio - theoretical max speedup
};

// Analyze a workload to determine theoretical metrics
TheoreticalMetrics analyze_workload_theory(const std::vector<Range> &ranges);

// Generate CSV with theoretical speedup for different processor counts
bool generate_theoretical_speedup_csv(
    const std::vector<std::vector<Range>> &workloads,
    const std::vector<std::string> &workload_descriptions,
    const std::string &output_filename);

#endif // THEORETICAL_ANALYSIS_H
