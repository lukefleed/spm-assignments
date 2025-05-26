// include/record.h
#ifndef RECORD_H
#define RECORD_H

#include <cstddef> // For size_t

// Maximum payload size for the rpayload array in struct Record.
// This should be at least as large as the maximum -r value (e.g., 256).
#ifndef MAX_RPAYLOAD_SIZE
#define MAX_RPAYLOAD_SIZE 256
#endif

struct Record {
  unsigned long key;                // Value used for sorting
  char rpayload[MAX_RPAYLOAD_SIZE]; // Fixed-size buffer for payload data
};

#endif // RECORD_H
