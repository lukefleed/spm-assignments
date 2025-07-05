#ifndef RECORD_HPP
#define RECORD_HPP

#include <cstddef> // used for size_t
#include <cstring>

/**
 * @brief Record structure for variable-size payload sorting
 *
 * Dynamic payload allocation enables runtime payload size configuration
 * via command-line parameters. Move semantics optimize container operations.
 */
struct Record {
  unsigned long key;   ///< Sorting key
  char *payload;       ///< Dynamic payload buffer
  size_t payload_size; ///< Payload size in bytes

  /**
   * @brief Initialize record with specified payload size
   * @param payload_size Payload allocation size in bytes
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

  /**
   * @brief Move constructor for Record class.
   *
   * Constructs a Record object by moving the contents from another Record
   * object. After the move operation, the source object is left in a valid but
   * unspecified state with its payload pointer set to nullptr and payload_size
   * set to 0.
   *
   * @param other The Record object to move from (rvalue reference)
   * @note The moved-from object's payload is set to nullptr to avoid double
   * deletion
   */
  Record(Record &&other) noexcept
      : key(other.key), payload(other.payload),
        payload_size(other.payload_size) {
    other.payload = nullptr;
    other.payload_size = 0;
  }

  /**
   * @brief Move assignment operator for Record class
   *
   * Transfers ownership of resources from another Record object to this one.
   * The source object is left in a valid but unspecified state with its
   * payload pointer set to nullptr to prevent double deletion.
   *
   * @param other The Record object to move from (rvalue reference)
   * @return Reference to this Record object after the move operation
   * @note This operation is noexcept and provides strong exception safety
   * @note Self-assignment is handled safely by checking pointer equality
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

  /**
   * @brief Destructor for the Record class.
   *
   * Cleans up dynamically allocated memory by deleting the payload array.
   * This prevents memory leaks when Record objects go out of scope or are
   * explicitly deleted.
   */
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
 * @brief Key-based comparator for sorting algorithms
 */
struct RecordComparator {
  bool operator()(const Record &a, const Record &b) const {
    return a.key < b.key;
  }
};

#endif // RECORD_HPP
