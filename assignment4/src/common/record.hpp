#ifndef RECORD_HPP
#define RECORD_HPP

#include <cstddef>
#include <cstring>

/**
 * @brief Fixed-size record structure for sorting
 */
struct Record {
  unsigned long key;   ///< Sorting key
  char *payload;       ///< Variable-size payload
  size_t payload_size; ///< Size of payload for copying

  /**
   * @brief Construct record with specified payload size
   */
  Record(size_t payload_size = 0)
      : key(0), payload(nullptr), payload_size(payload_size) {
    if (payload_size > 0) {
      payload = new char[payload_size];
      std::memset(payload, 0, payload_size);
    }
  }

  Record(const Record &) = delete;
  Record &operator=(const Record &) = delete;

  Record(Record &&other) noexcept
      : key(other.key), payload(other.payload),
        payload_size(other.payload_size) {
    other.payload = nullptr;
    other.payload_size = 0;
  }

  Record &operator=(Record &&other) noexcept {
    if (this != &other) {
      delete[] payload;
      key = other.key;
      payload = other.payload;
      payload_size = other.payload_size;
      other.payload = nullptr;
      other.payload_size = 0;
    }
    return *this;
  }

  ~Record() { delete[] payload; }

  // Comparison operators for sorting
  bool operator<(const Record &other) const { return key < other.key; }

  bool operator<=(const Record &other) const { return key <= other.key; }

  bool operator>(const Record &other) const { return key > other.key; }

  bool operator>=(const Record &other) const { return key >= other.key; }

  bool operator==(const Record &other) const { return key == other.key; }

  bool operator!=(const Record &other) const { return key != other.key; }
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
