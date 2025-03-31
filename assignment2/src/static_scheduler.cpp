#include "static_scheduler.h"
#include "collatz.h" // For collatz_steps and find_max_steps_in_subrange
#include <algorithm> // For std::min, std::max
#include <atomic>    // For std::atomic and memory orders
#include <cmath> // For std::min, std::max (though <algorithm> is more standard for these)
#include <iostream> // For verbose output
#include <thread>   // For std::thread, std::thread::hardware_concurrency
#include <vector>   // For std::vector

/**
 * @brief Worker function for the Static Block-Cyclic scheduling strategy.
 *
 * This function is executed by each participating thread. It processes blocks
 * of numbers (defined by `block_size`) from the `global_ranges` in a
 * round-robin fashion based on the thread's `thread_id`. This approach aims to
 * balance load by distributing consecutive blocks across different threads,
 * which can be beneficial if computational cost varies smoothly across the
 * number space.
 *
 * @param thread_id The unique identifier for this thread (0 to num_threads -
 * 1).
 * @param num_threads The total number of worker threads participating.
 * @param block_size The number of consecutive integers each block contains.
 * @param global_ranges A read-only reference to the vector of input ranges.
 * @param results A reference to the vector where computed maximum steps for
 * each range are stored atomically.
 * @param verbose If true, enables detailed diagnostic output.
 *
 * @note The atomicity for updating the maximum steps per range
 * (`results[range_idx].max_steps`) is crucial. `fetch_max` (C++20) or a
 * compare-and-swap (CAS) loop (pre-C++20) ensures that updates from concurrent
 * threads are correctly merged without data races. `memory_order_relaxed` is
 * often sufficient for `fetch_max` or the load in the CAS loop when the primary
 * goal is just to find the maximum value without enforcing strict ordering
 * relative to other memory operations, as the final result is only needed
 * after thread joins. `memory_order_release` on success in the CAS ensures
 * visibility to other threads if needed, while `memory_order_relaxed` on
 * failure is standard practice.
 */
void static_worker(int thread_id, int num_threads, ull block_size,
                   const std::vector<Range> &global_ranges,
                   std::vector<RangeResult> &results, bool verbose = false) {
  // Accumulates the total count of numbers processed in ranges preceding the
  // current one. Used to calculate the global index of a block across all
  // ranges.
  ull total_numbers_processed_in_prev_ranges = 0;

  // Prevent division by zero or infinite loops if block_size is invalid.
  if (block_size == 0) {
    if (verbose)
      std::cerr << "Thread " << thread_id
                << " received block_size=0, exiting worker." << std::endl;
    return;
  }

  // Iterate through each range provided in the input configuration.
  for (size_t range_idx = 0; range_idx < global_ranges.size(); ++range_idx) {
    const auto current_range = global_ranges[range_idx];

    // Skip ranges where the start is greater than the end (invalid or empty
    // range).
    if (current_range.start > current_range.end) {
      // This check prevents processing ill-defined ranges.
      continue;
    }

    // Calculate the total number of integers within the current range.
    ull range_len = current_range.end - current_range.start + 1;

    // Determine the number of blocks needed to cover this range.
    // Uses ceiling division `(N + D - 1) / D` to ensure the last partial block
    // is included.
    ull num_blocks_in_range = (range_len + block_size - 1) / block_size;

    // Track the maximum steps found by this thread *within the current range*.
    // Reset for each range to correctly update the per-range result.
    ull thread_local_max_for_range = 0;

    // Iterate through all blocks notionally belonging to the current range.
    for (ull block_in_range_idx = 0; block_in_range_idx < num_blocks_in_range;
         ++block_in_range_idx) {
      // Calculate the global index of this block across all ranges processed so
      // far. This ensures the round-robin assignment wraps correctly across
      // range boundaries.
      ull global_block_idx =
          (total_numbers_processed_in_prev_ranges / block_size) +
          block_in_range_idx;

      // Assign the block to a thread using the modulo operator (block-cyclic
      // distribution). Casting to ull avoids potential signed/unsigned
      // comparison warnings/issues.
      if ((global_block_idx % static_cast<ull>(num_threads)) ==
          static_cast<ull>(thread_id)) {

        // This thread is responsible for this block. Calculate its boundaries.
        ull block_start = current_range.start + block_in_range_idx * block_size;
        // Ensure the block end does not exceed the actual end of the current
        // range.
        ull block_end =
            std::min(current_range.end, block_start + block_size - 1);

        // Defensive check: ensure block start doesn't exceed block end after
        // std::min adjustment
        if (block_start > block_end)
          continue;

        if (verbose) {
          // Detailed logging for debugging assignment and processing.
          std::cout << "Thread " << thread_id << " processing block "
                    << global_block_idx << " (Range " << range_idx
                    << ", local block " << block_in_range_idx << ", ["
                    << block_start << "-" << block_end << "])" << std::endl;
        }

        // Perform the core Collatz computation for the assigned sub-range
        // (block).
        ull block_max_steps =
            find_max_steps_in_subrange(block_start, block_end);

        // Update the maximum step count found by this thread *within this
        // range*.
        thread_local_max_for_range =
            std::max(thread_local_max_for_range, block_max_steps);
      }
    }

    // After processing all blocks *assigned to this thread* within the current
    // range, atomically update the shared result for this range.
    if (thread_local_max_for_range > 0) {
// Use C++20 fetch_max if available for a more concise atomic maximum update.
#if defined(__cpp_lib_atomic_fetch_max) && __cpp_lib_atomic_fetch_max >= 202002L
      results[range_idx].max_steps.fetch_max(thread_local_max_for_range,
                                             std::memory_order_relaxed);
#else
      // Pre-C++20: Use a compare-exchange loop for atomic maximum update.
      ull current_max =
          results[range_idx].max_steps.load(std::memory_order_relaxed);
      // Loop until the update succeeds or the local max is no longer greater.
      while (thread_local_max_for_range > current_max) {
        if (results[range_idx].max_steps.compare_exchange_weak(
                current_max, thread_local_max_for_range,
                std::memory_order_release, // Ensures visibility on success
                std::memory_order_relaxed  // Sufficient on failure
                )) {
          break; // Update successful.
        }
        // CAS failed: current_max was updated by the call, retry the
        // comparison.
      }
#endif
    }

    // Add the length of the just-processed range to the accumulator for the
    // next iteration.
    total_numbers_processed_in_prev_ranges += range_len;
  } // End loop over ranges

  if (verbose) {
    std::cout << "Thread " << thread_id << " finished static_worker."
              << std::endl;
  }
}

/**
 * @brief Worker function for the Static Block scheduling strategy.
 *
 * Each thread is assigned a single, contiguous chunk of the total work within
 * each range. The range is divided as evenly as possible among the available
 * threads. This strategy minimizes synchronization overhead but can suffer from
 * load imbalance if the computational cost is unevenly distributed within the
 * range (e.g., some blocks take much longer).
 *
 * @param thread_id The unique identifier for this thread (0 to num_threads -
 * 1).
 * @param num_threads The total number of worker threads participating.
 * @param global_ranges A read-only reference to the vector of input ranges.
 * @param results A reference to the vector where computed maximum steps for
 * each range are stored atomically.
 * @param verbose If true, enables detailed diagnostic output.
 */
void static_block_worker(int thread_id, int num_threads,
                         const std::vector<Range> &global_ranges,
                         std::vector<RangeResult> &results,
                         bool verbose = false) {
  // Iterate through each range provided in the input configuration.
  for (size_t range_idx = 0; range_idx < global_ranges.size(); ++range_idx) {
    const auto current_range = global_ranges[range_idx];

    // Skip ranges where the start is greater than the end.
    if (current_range.start > current_range.end) {
      continue;
    }

    ull range_len = current_range.end - current_range.start + 1;

    // Calculate the base size of the block assigned to each thread using
    // integer division.
    ull base_block_size = range_len / num_threads;
    // Calculate the number of threads that will receive an extra element due to
    // remainder.
    ull remainder = range_len % num_threads;

    // Determine the start and end index for this specific thread's block.
    ull thread_start;
    ull thread_end;

    if (static_cast<ull>(thread_id) < remainder) {
      // Threads with id < remainder get one extra element.
      thread_start = current_range.start +
                     static_cast<ull>(thread_id) * (base_block_size + 1);
      thread_end = thread_start + base_block_size;
    } else {
      // Threads with id >= remainder get the base block size.
      // Start index needs to account for the larger blocks assigned to earlier
      // threads.
      thread_start =
          current_range.start + remainder * (base_block_size + 1) +
          (static_cast<ull>(thread_id) - remainder) * base_block_size;
      thread_end = thread_start + base_block_size - 1;
    }

    // Handle potential empty block assignment if range_len < num_threads
    if (thread_start > current_range.end || thread_start > thread_end) {
      if (verbose) {
        std::cout << "Thread " << thread_id
                  << " assigned empty block for range " << range_idx
                  << ", skipping." << std::endl;
      }
      continue; // No work for this thread in this range.
    }
    // Ensure thread_end does not exceed the actual range end (especially
    // important for the last thread).
    thread_end = std::min(thread_end, current_range.end);

    if (verbose) {
      std::cout << "Thread " << thread_id << " processing block ["
                << thread_start << "-" << thread_end << "] for range "
                << range_idx << std::endl;
    }

    // Process the assigned contiguous block.
    ull local_max_steps = find_max_steps_in_subrange(thread_start, thread_end);

    // Atomically update the maximum steps found for this range.
    if (local_max_steps > 0) {
#if defined(__cpp_lib_atomic_fetch_max) && __cpp_lib_atomic_fetch_max >= 202002L
      results[range_idx].max_steps.fetch_max(local_max_steps,
                                             std::memory_order_relaxed);
#else
      // Pre-C++20: Use compare-exchange loop.
      ull current_max =
          results[range_idx].max_steps.load(std::memory_order_relaxed);
      while (local_max_steps > current_max) {
        if (results[range_idx].max_steps.compare_exchange_weak(
                current_max, local_max_steps, std::memory_order_release,
                std::memory_order_relaxed)) {
          break; // Update successful.
        }
      }
#endif
    }
  } // End loop over ranges

  if (verbose) {
    std::cout << "Thread " << thread_id << " finished static_block_worker."
              << std::endl;
  }
}

/**
 * @brief Worker function for the Static Cyclic scheduling strategy.
 *
 * Each thread processes individual numbers from the ranges in a round-robin
 * fashion (thread 0 takes elements 0, N, 2N, ...; thread 1 takes 1, N+1, 2N+1,
 * ...). This strategy provides good load balancing at a fine granularity but
 * can suffer from poor cache performance due to non-contiguous memory access
 * patterns if the underlying computation had locality. For Collatz, locality is
 * less pronounced.
 *
 * @param thread_id The unique identifier for this thread (0 to num_threads -
 * 1).
 * @param num_threads The total number of worker threads participating.
 * @param global_ranges A read-only reference to the vector of input ranges.
 * @param results A reference to the vector where computed maximum steps for
 * each range are stored atomically.
 * @param verbose If true, enables detailed diagnostic output.
 */
void static_cyclic_worker(int thread_id, int num_threads,
                          const std::vector<Range> &global_ranges,
                          std::vector<RangeResult> &results,
                          bool verbose = false) {
  // Iterate through each range provided in the input configuration.
  for (size_t range_idx = 0; range_idx < global_ranges.size(); ++range_idx) {
    const auto current_range = global_ranges[range_idx];

    // Skip ranges where the start is greater than the end.
    if (current_range.start > current_range.end) {
      continue;
    }

    // Track the maximum steps found by this thread *within the current range*.
    ull thread_local_max_for_range = 0;

    // Determine the first number this thread should process in the current
    // range. Start from `current_range.start` and find the first number `num`
    // such that
    // `(num - current_range.start) % num_threads == thread_id`.
    ull start_offset = 0;
    if (current_range.start >
        0) { // Avoid potential issues with start = 0 if ever allowed
      start_offset = (static_cast<ull>(thread_id) -
                      (current_range.start % num_threads) + num_threads) %
                     num_threads;
    } else { // If start is 0 (or negative, though disallowed)
      start_offset = static_cast<ull>(thread_id);
    }
    ull first_num = current_range.start + start_offset;

    // Process numbers assigned to this thread in a cyclic manner.
    // Increment by `num_threads` to jump to the next number assigned to this
    // thread.
    for (ull num = first_num; num <= current_range.end; num += num_threads) {
      // Optional verbose logging, potentially throttled to avoid excessive
      // output. if (verbose && (num % 10000 == first_num % 10000)) { //
      // Throttle verbose output
      //   std::cout << "Thread " << thread_id << " processing number " << num
      //             << " in range " << range_idx << std::endl;
      // }

      // Perform the core Collatz computation for the single assigned number.
      ull steps = collatz_steps(num);

      // Update the maximum step count found by this thread *within this range*.
      thread_local_max_for_range = std::max(thread_local_max_for_range, steps);
    }

    // Atomically update the shared result for this range after checking all
    // assigned numbers.
    if (thread_local_max_for_range > 0) {
#if defined(__cpp_lib_atomic_fetch_max) && __cpp_lib_atomic_fetch_max >= 202002L
      results[range_idx].max_steps.fetch_max(thread_local_max_for_range,
                                             std::memory_order_relaxed);
#else
      // Pre-C++20: Use compare-exchange loop.
      ull current_max =
          results[range_idx].max_steps.load(std::memory_order_relaxed);
      while (thread_local_max_for_range > current_max) {
        if (results[range_idx].max_steps.compare_exchange_weak(
                current_max, thread_local_max_for_range,
                std::memory_order_release, std::memory_order_relaxed)) {
          break; // Update successful.
        }
      }
#endif
    }
  } // End loop over ranges

  if (verbose) {
    std::cout << "Thread " << thread_id << " finished static_cyclic_worker."
              << std::endl;
  }
}

/**
 * @brief Main function to execute Collatz computation using a specified static
 * scheduling strategy.
 *
 * This function sets up the necessary data structures, selects the appropriate
 * worker function based on `config.static_variant`, creates and manages worker
 * threads, and ensures all threads complete before returning.
 *
 * @param config Configuration object containing scheduling parameters (variant,
 * thread count, chunk size), input ranges, and verbosity settings.
 * @param[out] results_out A vector to be populated with the computed
 * RangeResult objects. Any existing content will be cleared.
 * @return true if the execution setup and thread management were successful,
 * false otherwise (e.g., invalid config).
 */
bool run_static_scheduling(const Config &config,
                           std::vector<RangeResult> &results_out) {
  // --- Input Validation ---
  // Ensure a positive number of threads are requested.
  if (config.num_threads <= 0) {
    std::cerr
        << "Error: Number of threads must be positive for static scheduling."
        << std::endl;
    return false;
  }
  // Ensure a positive chunk size is provided if using block-cyclic variant.
  if (config.static_variant == StaticVariant::BLOCK_CYCLIC &&
      config.chunk_size == 0) {
    std::cerr << "Error: Static block-cyclic scheduling requires a positive "
                 "chunk size."
              << std::endl;
    return false;
  }
  // Block and Cyclic variants do not use chunk_size, so no check needed for
  // them.

  // --- Thread Pool Setup ---
  // Determine the actual number of threads to use. It's often beneficial
  // to not exceed the number of hardware threads available to avoid excessive
  // context switching overhead. Using 1 thread implies sequential execution
  // handled elsewhere. Note: config.num_threads should be > 1 if this function
  // is called. If config.num_threads == 1, the caller should use the sequential
  // implementation directly. However, we still handle it gracefully here.
  int threads_to_use =
      (config.num_threads == 1)
          ? 1
          : std::min(static_cast<int>(config.num_threads),
                     static_cast<int>(std::thread::hardware_concurrency()));
  // If user explicitly asked for more threads than hardware_concurrency, using
  // hardware_concurrency is generally a safer default for performance, though
  // allowing oversubscription could be an option.
  if (config.verbose && static_cast<int>(config.num_threads) > threads_to_use &&
      config.num_threads > 1) {
    std::cout << "Warning: Requested " << config.num_threads
              << " threads, but limiting to hardware concurrency of "
              << threads_to_use << " for static scheduling." << std::endl;
  }
  if (threads_to_use <=
      0) { // Should not happen if config.num_threads > 0, but defensive check.
    threads_to_use = 1;
  }

  std::vector<std::thread> threads;
  results_out.clear(); // Ensure the output vector is empty before population.

  // Initialize the results vector. One RangeResult per input range,
  // with the atomic max_steps initialized to 0.
  results_out.reserve(config.ranges.size()); // Pre-allocate memory
  for (const auto &r : config.ranges) {
    results_out.emplace_back(
        r); // Creates RangeResult with original range and max_steps=0
  }

  if (config.verbose) {
    std::string variant_name;
    switch (config.static_variant) {
    case StaticVariant::BLOCK:
      variant_name = "Block";
      break;
    case StaticVariant::CYCLIC:
      variant_name = "Cyclic";
      break;
    case StaticVariant::BLOCK_CYCLIC:
      variant_name = "Block-Cyclic";
      break;
    default:
      variant_name = "Unknown";
      break;
    }
    std::cout << "Starting static " << variant_name << " scheduling with "
              << threads_to_use << " threads.";
    // Only mention block size if relevant to the chosen variant.
    if (config.static_variant == StaticVariant::BLOCK_CYCLIC) {
      std::cout << " Block size: " << config.chunk_size;
    }
    std::cout << std::endl;
  }

  // Handle the "sequential" case where only 1 thread is used (or requested).
  // Although typically handled by the caller, running it here ensures
  // correctness if called with T=1.
  if (threads_to_use == 1) {
    if (config.verbose)
      std::cout << "Executing sequentially within run_static_scheduling "
                   "(num_threads=1)."
                << std::endl;
    // Call the appropriate worker directly on the main thread.
    switch (config.static_variant) {
    case StaticVariant::BLOCK:
      static_block_worker(0, 1, config.ranges, results_out, config.verbose);
      break;
    case StaticVariant::CYCLIC:
      static_cyclic_worker(0, 1, config.ranges, results_out, config.verbose);
      break;
    case StaticVariant::BLOCK_CYCLIC:
      // Block-Cyclic with 1 thread and any block_size is equivalent to Block.
      // Call static_worker which handles this correctly (global_block_idx % 1
      // == 0).
      static_worker(0, 1, config.chunk_size, config.ranges, results_out,
                    config.verbose);
      break;
    }
    if (config.verbose)
      std::cout << "Static scheduling (sequential mode) finished." << std::endl;
    return true; // Sequential execution completed.
  }

  // --- Thread Creation and Dispatch ---
  // Launch the worker threads, passing arguments appropriately.
  // Use std::cref for read-only data (config.ranges) to avoid copying.
  // Use std::ref for mutable data (results_out) that threads need to modify.
  threads.reserve(threads_to_use); // Pre-allocate space for thread objects.
  switch (config.static_variant) {
  case StaticVariant::BLOCK:
    for (int i = 0; i < threads_to_use; ++i) {
      threads.emplace_back(static_block_worker, i, threads_to_use,
                           std::cref(config.ranges), std::ref(results_out),
                           config.verbose);
    }
    break;
  case StaticVariant::CYCLIC:
    for (int i = 0; i < threads_to_use; ++i) {
      threads.emplace_back(static_cyclic_worker, i, threads_to_use,
                           std::cref(config.ranges), std::ref(results_out),
                           config.verbose);
    }
    break;
  case StaticVariant::BLOCK_CYCLIC:
    // Ensure chunk_size is valid (checked earlier, but reinforce).
    if (config.chunk_size == 0)
      return false; // Should not happen due to earlier check
    for (int i = 0; i < threads_to_use; ++i) {
      threads.emplace_back(static_worker, i, threads_to_use, config.chunk_size,
                           std::cref(config.ranges), std::ref(results_out),
                           config.verbose);
    }
    break;
  // Default case should not be reachable if config validation is robust.
  default:
    std::cerr << "Error: Invalid static scheduling variant in dispatch."
              << std::endl;
    return false;
  }

  // --- Synchronization ---
  // Wait for all launched worker threads to complete their execution.
  // Joining ensures that all results are computed and memory operations
  // (especially the final atomic updates) are synchronized before proceeding.
  for (auto &t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }

  if (config.verbose) {
    std::cout << "Static scheduling finished." << std::endl;
  }

  return true; // Indicate successful completion.
}

/**
 * @brief Legacy function wrapper for static block-cyclic scheduling.
 *
 * This function exists potentially for backward compatibility or specific use
 * cases. It ensures the BLOCK_CYCLIC variant is selected and delegates to the
 * main `run_static_scheduling` function.
 *
 * @param config Configuration object. The `static_variant` field will be
 * ignored and forced to BLOCK_CYCLIC.
 * @param results_out Output vector for results.
 * @return bool Result of the delegated call to `run_static_scheduling`.
 * @deprecated Prefer using `run_static_scheduling` directly with the
 * appropriate `static_variant` set in the Config.
 */
bool run_static_block_cyclic(const Config &config,
                             std::vector<RangeResult> &results_out) {
  // Create a mutable copy of the config to explicitly set the variant.
  Config block_cyclic_config = config;
  block_cyclic_config.static_variant = StaticVariant::BLOCK_CYCLIC;

  // Delegate to the primary static scheduling implementation.
  return run_static_scheduling(block_cyclic_config, results_out);
}
