/**
 * @file csv_format.h
 * @brief Standardized CSV format for all performance tests
 */

#pragma once

#include <fstream>
#include <iomanip>
#include <string>

namespace csv_format {

/**
 * @brief Standardized CSV header for single-node performance tests
 */
inline void write_single_node_csv_header(std::ofstream &file) {
  file << "Test_Type,Implementation,Data_Size,Payload_Size_Bytes,Threads,"
       << "Execution_Time_ms,Throughput_MRec_per_sec,Speedup_vs_StdSort,"
       << "Speedup_vs_Sequential,Efficiency_Percent,Valid\n";
}

/**
 * @brief Write single-node performance result to CSV
 */
inline void write_single_node_csv_row(
    std::ofstream &file, const std::string &test_type,
    const std::string &implementation, size_t data_size,
    size_t payload_size_bytes, size_t threads, double execution_time_ms,
    double throughput_mrec_per_sec, double speedup_vs_std_sort,
    double speedup_vs_sequential, double efficiency_percent, bool valid) {
  file << test_type << "," << implementation << "," << data_size << ","
       << payload_size_bytes << "," << threads << "," << std::fixed
       << std::setprecision(3) << execution_time_ms << ","
       << std::setprecision(3) << throughput_mrec_per_sec << ","
       << std::setprecision(3) << speedup_vs_std_sort << ","
       << std::setprecision(3) << speedup_vs_sequential << ","
       << std::setprecision(1) << efficiency_percent << ","
       << (valid ? "true" : "false") << "\n";
}

/**
 * @brief Standardized CSV header for hybrid performance tests
 */
inline void write_hybrid_csv_header(std::ofstream &file) {
  file << "Test_Type,Implementation,Data_Size,Payload_Size_Bytes,MPI_Processes,"
       << "FF_Threads_Per_Process,Total_Threads,Execution_Time_ms,"
       << "Throughput_MRec_per_sec,Speedup_vs_Parallel_Baseline,"
       << "MPI_Efficiency_Percent,Total_Efficiency_Percent,Valid\n";
}

/**
 * @brief Write hybrid performance result to CSV
 */
inline void write_hybrid_csv_row(
    std::ofstream &file, const std::string &test_type,
    const std::string &implementation, size_t data_size,
    size_t payload_size_bytes, int mpi_processes, int ff_threads_per_process,
    int total_threads, double execution_time_ms, double throughput_mrec_per_sec,
    double speedup_vs_parallel_baseline, double mpi_efficiency_percent,
    double total_efficiency_percent, bool valid) {
  file << test_type << "," << implementation << "," << data_size << ","
       << payload_size_bytes << "," << mpi_processes << ","
       << ff_threads_per_process << "," << total_threads << "," << std::fixed
       << std::setprecision(3) << execution_time_ms << ","
       << std::setprecision(3) << throughput_mrec_per_sec << ","
       << std::setprecision(3) << speedup_vs_parallel_baseline << ","
       << std::setprecision(1) << mpi_efficiency_percent << ","
       << std::setprecision(1) << total_efficiency_percent << ","
       << (valid ? "true" : "false") << "\n";
}

/**
 * @brief Generate descriptive filename for test results
 */
inline std::string
generate_results_filename(const std::string &test_type,
                          const std::string &date_suffix = "") {
  std::string filename = "results_" + test_type;
  if (!date_suffix.empty()) {
    filename += "_" + date_suffix;
  }
  filename += ".csv";
  return filename;
}

} // namespace csv_format
