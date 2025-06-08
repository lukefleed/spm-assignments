#ifndef RECORD_HPP
#define RECORD_HPP

#include <cstddef>
#include <cstring>

/**
 * @brief Record structure with variable-size payload for high-performance
 * sorting
 *
 * Uses raw pointer for payload to minimize overhead during sort operations.
 * Copy operations disabled to prevent expensive deep copies during sorting.
 */
struct Record {
  unsigned long key;   ///< Primary sorting key
  char *payload;       ///< Variable-size payload data
  size_t payload_size; ///< Payload byte count

  /**
   * @brief Construct record with zero-initialized payload
   * @param payload_size Payload allocation size in bytes
   */
  Record(size_t payload_size = 0)
      : key(0), payload(nullptr), payload_size(payload_size) {
    if (payload_size > 0) {
      payload = new char[payload_size];
      std::memset(payload, 0, payload_size);
    }
  }

  /// Copy operations disabled to prevent expensive deep copies during sorting
  Record(const Record &) = delete;
  Record &operator=(const Record &) = delete;

  /**
   * @brief Move constructor for efficient container operations
   */
  Record(Record &&other) noexcept
      : key(other.key), payload(other.payload),
        payload_size(other.payload_size) {
    other.payload = nullptr;
    other.payload_size = 0;
  }

  /**
   * @brief Move assignment for efficient container operations
   */
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

  /// Destructor releases allocated payload memory
  ~Record() { delete[] payload; }

  /// @name Comparison operators for key-based sorting
  /// @{
  bool operator<(const Record &other) const { return key < other.key; }
  bool operator<=(const Record &other) const { return key <= other.key; }
  bool operator>(const Record &other) const { return key > other.key; }
  bool operator>=(const Record &other) const { return key >= other.key; }
  bool operator==(const Record &other) const { return key == other.key; }
  bool operator!=(const Record &other) const { return key != other.key; }
  /// @}
};

/**
 * @brief Function object for key-based record comparison
 *
 * Provides strict weak ordering for sorting algorithms.
 */
struct RecordComparator {
  bool operator()(const Record &a, const Record &b) const {
    return a.key < b.key;
  }
};

#endif // RECORD_HPP
