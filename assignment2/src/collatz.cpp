#include "collatz.h"
#include <algorithm> // For std::max
#include <limits>    // For std::numeric_limits
#include <stdexcept> // For std::overflow_error
#include <iostream>  // For std::cerr

/**
 * @brief Calculates the number of steps required for a given positive integer
 *        to reach 1 by following the Collatz sequence rules.
 *
 * The Collatz sequence is defined as:
 * - If n is even, the next term is n / 2.
 * - If n is odd, the next term is 3 * n + 1.
 * The conjecture posits that this sequence eventually reaches 1 for any
 * positive integer starting value.
 *
 * @param n The starting positive integer (must be > 0).
 * @return The number of steps taken to reach 1. Returns 0 if n is 0 or 1.
 * @throws std::overflow_error If the 3*n + 1 operation would exceed the
 *         maximum value representable by unsigned long long (ull).
 *
 * @note Includes an optimization for powers of 2 and uses bitwise operations
 *       for efficiency. The power-of-2 optimization leverages the GCC/Clang
 *       intrinsic `__builtin_ctzll` (count trailing zeros) for performance.
 *       A portable alternative could be implemented if needed.
 */
ull collatz_steps(ull n) {
  // Base cases: The sequence terminates immediately for 0 and 1.
  // Collatz is typically defined for positive integers, so n=0 yields 0 steps.
  if (n <= 1) {
    return 0;
  }

  ull steps = 0;
  while (n != 1) {
    // Optimization: Check if n is a power of 2.
    // `(n & (n - 1)) == 0` is true if and only if n is 0 or a power of 2.
    // Since we've already handled n=0 and n=1, this condition means n is 2, 4,
    // 8, ...
    if ((n & (n - 1)) == 0 && n != 0) {
      // If n is a power of 2 (say 2^k), it will take exactly k steps (repeated
      // divisions by 2) to reach 1. The number of steps k is equal to the
      // number of trailing zeros in the binary representation of n (log2(n)).
      // `__builtin_ctzll` efficiently computes this for unsigned long long.
      // This avoids performing k individual division steps in the loop.
#ifdef __GNUC__ // Check if using GCC or Clang which support this intrinsic
      return steps + __builtin_ctzll(n);
#else
      // Portable fallback (less efficient): Continue the loop normally.
      // Or implement a portable count_trailing_zeros function.
      // For simplicity, we'll just let the loop handle it if the intrinsic
      // isn't available.
#endif
    }

    // Check if n is even using a bitwise AND operation (n & 1 == 0).
    // This is typically faster than the modulo operator (n % 2 == 0).
    if ((n & 1) == 0) {
      // n is even: divide by 2 using a right bit-shift (n >>= 1).
      // This is often faster than integer division (n /= 2) on many
      // architectures.
      n >>= 1;
    } else {
      // n is odd: calculate 3*n + 1.
      // Before performing the multiplication, check for potential overflow.
      // If n > (MAX_ULL - 1) / 3, then 3*n will exceed MAX_ULL - 1,
      // and adding 1 will certainly overflow.
      // Using `std::numeric_limits<ull>::max()` provides the maximum value for
      // ull.
      if (n > (std::numeric_limits<ull>::max() - 1) / 3) {
        // Throwing an exception halts computation cleanly upon detecting
        // overflow, preventing incorrect results due to wrap-around behavior.
        throw std::overflow_error(
            "Overflow detected in Collatz step (3*n + 1)");
      }
      n = 3 * n + 1;
    }
    // Increment the step counter after applying the Collatz rule.
    steps++;
  }
  // Once n reaches 1, the loop terminates, and the total steps are returned.
  return steps;
}

/**
 * @brief Finds the maximum number of Collatz steps for any integer within a
 * specified range.
 *
 * Iterates through each number `i` in the range [start, end] (inclusive),
 * calculates its Collatz steps using `collatz_steps(i)`, and returns the
 * highest step count encountered.
 *
 * @param start The starting integer of the range (inclusive).
 * @param end The ending integer of the range (inclusive).
 * @return The maximum number of Collatz steps found within the range.
 *         Returns 0 if the range is invalid (start > end) or empty after
 * adjusting start=0 to start=1.
 * @note Handles potential overflow within the `collatz_steps` function by
 * catching exceptions (if not caught earlier). Adjusts start = 0 to start = 1
 * as Collatz is defined for positive integers.
 */
ull find_max_steps_in_subrange(ull start, ull end) {
  // Handle the case where the provided range is logically invalid or empty.
  if (start > end) {
    return 0;
  }

  // Ensure the starting point is at least 1, as Collatz is defined for positive
  // integers. If the original start was 0, adjusting it to 1 maintains the
  // spirit of the calculation.
  if (start == 0) {
    start = 1;
    // If the original range was just [0, 0], adjusting start to 1 makes it [1,
    // 0], which will be correctly handled by the start > end check below (or
    // implicitly by the loop condition).
    if (start > end)
      return 0; // Handle the case where original range was [0,0]
  }

  ull max_s = 0; // Initialize maximum steps found so far to 0.

  // Iterate through every number from start to end (inclusive).
  // The loop condition `i <= end` correctly handles the iteration.
  for (ull i = start;; ++i) { // Loop structure changed slightly to handle i ==
                              // MAX condition inside
    try {
      ull current_steps = collatz_steps(i);
      // Update the maximum if the current number yields more steps.
      // Using std::max is clear and potentially optimized by the compiler.
      max_s = std::max(max_s, current_steps);
    } catch (const std::overflow_error &e) {
      // If collatz_steps throws an overflow error for a number `i` in the
      // range, we cannot determine the true maximum steps for that number or
      // subsequent ones. Options:
      // 1. Re-throw the exception to signal failure for the whole subrange.
      // 2. Log the error and return the maximum found *so far*.
      // 3. Log the error and return a special value (e.g., MAX_ULL) to indicate
      // error. Current approach: Log error and return max found so far, as some
      // results might still be useful.
      std::cerr << "Warning: Overflow occurred calculating steps for " << i
                << " within range [" << start << ", " << end << "]. "
                << "Result might be incomplete. Error: " << e.what()
                << std::endl;
      // Stop processing this subrange as results might be compromised.
      return max_s;
    }

    // Check for loop termination *after* processing `i`.
    // This ensures that `end` itself is processed.
    // Also, handle the edge case where `i` reaches MAX_ULL to prevent overflow
    // on `++i`.
    if (i == end || i == std::numeric_limits<ull>::max()) {
      break;
    }
  }
  // Return the highest step count found across the valid portion of the range.
  return max_s;
}
