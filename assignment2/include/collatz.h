#ifndef COLLATZ_H
#define COLLATZ_H

#include "common_types.h" // Defines ull (unsigned long long)

/**
 * @brief Calculates the number of steps in the Collatz sequence for n.
 * @param n The starting positive integer.
 * @return The number of steps to reach 1.
 * @throws std::overflow_error If an arithmetic overflow occurs during
 * calculation.
 */
ull collatz_steps(ull n);

/**
 * @brief Finds the maximum Collatz steps within the range [start, end].
 * @param start The beginning of the range (inclusive).
 * @param end The end of the range (inclusive).
 * @return The maximum step count found, or 0 if start > end or calculation
 * fails due to overflow.
 */
ull find_max_steps_in_subrange(ull start, ull end);

#endif // COLLATZ_H
