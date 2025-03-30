#include "collatz.h"
#include <limits>    // For std::numeric_limits
#include <stdexcept> // For std::overflow_error

/**
 * @brief Calculates the number of steps in the Collatz sequence for a given
 * number
 *
 * The Collatz conjecture states that repeatedly applying the following
 * operations will eventually reduce any positive integer to 1:
 * - If n is even, divide it by 2
 * - If n is odd, multiply by 3 and add 1
 *
 * @param n The starting number for the Collatz sequence
 * @return The number of steps required to reach 1
 * @throws std::overflow_error If arithmetic overflow would occur during
 * calculation
 */
ull collatz_steps(ull n) {
  if (n <= 1) {
    return 0;
  }

  ull steps = 0;
  while (n != 1) {
    // Optimization: if n is a power of 2, we can directly calculate remaining
    // steps
    if ((n & (n - 1)) == 0) {
      // n is a power of 2, add trailing zeros (equivalent to log2(n))
      // Note: __builtin_ctzll is a GNU compiler intrinsic that counts trailing
      // zeros This is used for performance but could be replaced with a
      // portable implementation if needed
      return steps + __builtin_ctzll(n);
    }

    if ((n & 1) == 0) {
      // n is even
      n >>= 1; // Faster than n /= 2
    } else {
      // Check for potential overflow before multiplying
      if (n > (std::numeric_limits<ull>::max() - 1) / 3) {
        throw std::overflow_error("Arithmetic overflow in Collatz calculation");
      }
      n = 3 * n + 1;
    }
    steps++;
  }
  return steps;
}

/**
 * @brief Finds the maximum number of Collatz steps for any number in the given
 * range
 *
 * Iterates through each number in the range [start, end] and calculates
 * the number of steps in its Collatz sequence, tracking the maximum.
 *
 * @param start The lower bound of the range (inclusive)
 * @param end The upper bound of the range (inclusive)
 * @return The maximum number of Collatz steps found in the range
 */
ull find_max_steps_in_subrange(ull start, ull end) {
  // Handle invalid range cases
  if (start > end) {
    return 0; // Invalid or empty range
  }

  // Ensure we start from at least 1 (0 is not valid for Collatz)
  if (start == 0) {
    start = 1;
  }

  ull max_s = 0;

  for (ull i = start; i <= end; ++i) {
    ull current_steps = collatz_steps(i);
    if (current_steps > max_s) {
      max_s = current_steps;
    }

    // Prevent overflow when incrementing if i is at the maximum possible value
    if (i == std::numeric_limits<ull>::max()) {
      break;
    }
  }
  return max_s;
}
