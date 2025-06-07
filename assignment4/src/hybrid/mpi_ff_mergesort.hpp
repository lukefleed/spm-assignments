// FILE: src/hybrid/mpi_ff_mergesort.hpp

#pragma once

#include "../common/record.hpp"
#include <memory>
#include <mpi.h>
#include <string>
#include <vector>

namespace hybrid {

struct HybridConfig {
  size_t parallel_threads{0};
  size_t min_local_threshold{10000};
};

struct HybridMetrics {
  double total_time{0.0};
  double local_sort_time{0.0};
  double merge_time{0.0};
  double communication_time{0.0};
  size_t bytes_communicated{0};
  size_t local_elements{0};
};

class HybridMergeSort {
public:
  explicit HybridMergeSort(const HybridConfig &config);
  ~HybridMergeSort();

  HybridMergeSort(const HybridMergeSort &) = delete;
  HybridMergeSort &operator=(const HybridMergeSort &) = delete;
  HybridMergeSort(HybridMergeSort &&) = default;
  HybridMergeSort &operator=(HybridMergeSort &&) = default;

  std::vector<Record> sort(std::vector<Record> &data, size_t payload_size);
  const HybridMetrics &get_metrics() const { return metrics_; }

private:
  void distribute_data(std::vector<Record> &local_data,
                       const std::vector<Record> &global_data);
  void sort_local_data(std::vector<Record> &data);
  void hierarchical_merge(std::vector<Record> &local_data);

  std::vector<Record> merge_with_partner(std::vector<Record> &local_data,
                                         int partner_rank, bool is_receiver,
                                         MPI_Comm comm);

  void update_metrics(const std::string &phase, double elapsed_time);

  static std::vector<char> pack_records(const std::vector<Record> &records,
                                        size_t record_byte_size);
  static std::vector<Record> unpack_records(const std::vector<char> &buffer,
                                            size_t payload_size,
                                            size_t record_byte_size);
  static std::vector<Record> merge_sorted_vectors(std::vector<Record> &left,
                                                  std::vector<Record> &right);

  HybridConfig config_;
  int mpi_rank_;
  int mpi_size_;
  size_t payload_size_;
  HybridMetrics metrics_;
};

} // namespace hybrid
