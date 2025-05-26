#ifndef RECORD_H
#define RECORD_H

#include <cstddef> // For size_t

// Define the maximum possible payload size.
// This must be a compile-time constant.
// The actual payload size used for a given run (r_payload_size_bytes)
// will be <= MAX_RPAYLOAD_SIZE.
#ifndef MAX_RPAYLOAD_SIZE
#define MAX_RPAYLOAD_SIZE                                                      \
  256 // Default, can be changed or set via compile flags
#endif

struct Record {
  unsigned long key;                // Sorting value
  char rpayload[MAX_RPAYLOAD_SIZE]; // Fixed-size buffer for payload
                                    // The actual used part is
                                    // r_payload_size_bytes
};

#endif // RECORD_H
