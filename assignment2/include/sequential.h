#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include "common_types.h"
#include <vector>

/**
 * @brief Calculates the maximum number of Collatz steps sequentially for each
 * range.
 * @param ranges A vector of input ranges.
 * @return A vector containing the maximum number of steps for each
 * corresponding range.
 */
std::vector<ull> run_sequential(const std::vector<Range> &ranges);

#endif // SEQUENTIAL_H
