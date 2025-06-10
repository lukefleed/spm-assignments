#ifndef RECORD_HPP
#define RECORD_HPP

#include "../include/config.hpp"
#include <cstddef>
#include <cstring>

/**
 * @brief Record structure with fixed-size payload for high-performance sorting
 *
 * Conforms to assignment specification with contiguous memory layout for
 * efficient MPI communication and cache performance.
 */
struct Record {
  unsigned long key;       ///< Primary sorting key
  char rpayload[RPAYLOAD]; ///< Fixed-size payload data

  /**
   * @brief Default constructor with zero-initialized payload
   */
  Record() : key(0) { std::memset(rpayload, 0, RPAYLOAD); }

  /**
   * @brief Constructor with key initialization
   * @param k Initial key value
   */
  explicit Record(unsigned long k) : key(k) {
    std::memset(rpayload, 0, RPAYLOAD);
  }

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
