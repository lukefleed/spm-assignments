#ifndef RECORD_HPP
#define RECORD_HPP

#include <cstddef>
#include <cstring>

/**
 * @brief Fixed-size record structure for sorting
 */
struct Record {
  unsigned long key; ///< Sorting key
  char *payload;     ///< Variable-size payload

  /**
   * @brief Construct record with specified payload size
   */
  Record(size_t payload_size = 0) : key(0), payload(nullptr) {
    if (payload_size > 0) {
      payload = new char[payload_size];
      std::memset(payload, 0, payload_size);
    }
  }

  Record(const Record &) = delete;
  Record &operator=(const Record &) = delete;

  Record(Record &&other) noexcept : key(other.key), payload(other.payload) {
    other.payload = nullptr;
  }

  Record &operator=(Record &&other) noexcept {
    if (this != &other) {
      delete[] payload;
      key = other.key;
      payload = other.payload;
      other.payload = nullptr;
    }
    return *this;
  }

  ~Record() { delete[] payload; }
};

/**
 * @brief Comparator for sorting by key
 */
struct RecordComparator {
  bool operator()(const Record &a, const Record &b) const {
    return a.key < b.key;
  }
};

#endif // RECORD_HPP
