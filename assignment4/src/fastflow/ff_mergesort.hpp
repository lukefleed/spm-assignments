#ifndef FF_MERGESORT_HPP
#define FF_MERGESORT_HPP

#include "../common/record.hpp"
#include <memory>
#include <vector>

/**
 * @brief Task structure for FastFlow pipeline
 */
struct SortTask {
  std::vector<std::unique_ptr<Record>> *data;
  size_t start;
  size_t end;
  size_t chunk_id;
  size_t total_chunks;

  SortTask(std::vector<std::unique_ptr<Record>> *d, size_t s, size_t e,
           size_t id, size_t total)
      : data(d), start(s), end(e), chunk_id(id), total_chunks(total) {}
};

/**
 * @brief Result structure for sorted chunks
 */
struct SortedChunk {
  size_t start;
  size_t end;
  size_t chunk_id;
  size_t total_chunks;
};

#endif // FF_MERGESORT_HPP
