#include "static_scheduler.h"
#include "collatz.h"
#include <atomic>
#include <cmath>
#include <iostream>
#include <thread>
#include <vector>

/**
 * @brief Worker function executed by each thread in the static block-cyclic
 * scheduler
 *
 * This function implements a block-cyclic work distribution pattern where each
 * thread processes blocks of data in a round-robin fashion based on its thread
 * ID.
 *
 * @param thread_id ID of the current thread (0 to num_threads-1)
 * @param num_threads Total number of threads in the computation
 * @param block_size Size of each block for processing (chunk size)
 * @param global_ranges Vector of ranges to process (passed by const reference)
 * @param results Vector to store computation results (passed by reference for
 * updating)
 * @param verbose Optional flag for debug output
 */
void static_worker(int thread_id, int num_threads, ull block_size,
                   const std::vector<Range> &global_ranges,
                   std::vector<RangeResult> &results, bool verbose = false) {
  // Track total numbers processed to calculate global block indices
  ull total_numbers_processed_in_prev_ranges = 0;

  if (block_size == 0) {
    // Guard against invalid block size to prevent division by zero
    return;
  }

  for (size_t range_idx = 0; range_idx < global_ranges.size(); ++range_idx) {
    const auto &current_range = global_ranges[range_idx];

    // Skip invalid ranges where start > end
    if (current_range.start > current_range.end) {
      continue;
    }

    ull range_len = current_range.end - current_range.start + 1;

    // Calculate number of blocks in this range using ceiling division
    ull num_blocks_in_range = (range_len + block_size - 1) / block_size;

    // Track local maximum steps for the current thread
    ull thread_local_max = 0;

    // Iterate over all blocks in the current range
    for (ull block_in_range_idx = 0; block_in_range_idx < num_blocks_in_range;
         ++block_in_range_idx) {
      // Calculate global block index across all ranges
      ull global_block_idx =
          total_numbers_processed_in_prev_ranges / block_size +
          block_in_range_idx;

      // Block-cyclic assignment: each thread processes blocks in round-robin
      // order Cast to same type to avoid signed/unsigned comparison issues
      if (global_block_idx % static_cast<ull>(num_threads) ==
          static_cast<ull>(thread_id)) {
        // Calculate the start and end of this block
        ull block_start = current_range.start + block_in_range_idx * block_size;
        ull block_end =
            std::min(current_range.end, block_start + block_size - 1);

        if (verbose) {
          std::cout << "Thread " << thread_id << " processing block "
                    << global_block_idx << " (Range " << range_idx
                    << ", local block " << block_in_range_idx << ", ["
                    << block_start << "-" << block_end << "])" << std::endl;
        }

        // Compute maximum steps for this block
        ull local_max_steps =
            find_max_steps_in_subrange(block_start, block_end);

        // Update thread-local maximum steps
        thread_local_max = std::max(thread_local_max, local_max_steps);
      }
    }

    // Atomically update the maximum steps for this range at the end of range
    // processing
    if (thread_local_max > 0) {
#if __cplusplus >= 202002L
      results[range_idx].max_steps.fetch_max(thread_local_max,
                                             std::memory_order_relaxed);
#else
      ull current_max =
          results[range_idx].max_steps.load(std::memory_order_relaxed);
      while (thread_local_max > current_max) {
        if (results[range_idx].max_steps.compare_exchange_weak(
                current_max, thread_local_max, std::memory_order_release,
                std::memory_order_relaxed)) {
          break; // Update successful
        }
        // Update failed, current_max was updated by another thread, retry
        // with new value
      }
#endif
    }

    // Update global counter for the next range
    total_numbers_processed_in_prev_ranges += range_len;
  }

  if (verbose)
    std::cout << "Thread " << thread_id << " finished." << std::endl;
}

/**
 * @brief Worker function for block scheduling
 *
 * Each thread processes a single contiguous block of the range.
 * Work is divided evenly among threads.
 */
void static_block_worker(int thread_id, int num_threads,
                         const std::vector<Range> &global_ranges,
                         std::vector<RangeResult> &results,
                         bool verbose = false) {
  for (size_t range_idx = 0; range_idx < global_ranges.size(); ++range_idx) {
    const auto &current_range = global_ranges[range_idx];

    // Skip invalid ranges where start > end
    if (current_range.start > current_range.end) {
      continue;
    }

    ull range_len = current_range.end - current_range.start + 1;

    // Calculate the size of each block per thread
    ull block_size = range_len / num_threads;

    // Calculate this thread's start and end
    ull thread_start = current_range.start + thread_id * block_size;
    ull thread_end;

    // The last thread takes any remaining elements
    if (thread_id == num_threads - 1) {
      thread_end = current_range.end;
    } else {
      thread_end = thread_start + block_size - 1;
    }

    if (verbose) {
      std::cout << "Thread " << thread_id << " processing block ["
                << thread_start << "-" << thread_end << "] for range "
                << range_idx << std::endl;
    }

    // Process the assigned block
    ull local_max_steps = find_max_steps_in_subrange(thread_start, thread_end);

    // Update the result atomically
    if (local_max_steps > 0) {
#if __cplusplus >= 202002L
      results[range_idx].max_steps.fetch_max(local_max_steps,
                                             std::memory_order_relaxed);
#else
      ull current_max =
          results[range_idx].max_steps.load(std::memory_order_relaxed);
      while (local_max_steps > current_max) {
        if (results[range_idx].max_steps.compare_exchange_weak(
                current_max, local_max_steps, std::memory_order_release,
                std::memory_order_relaxed)) {
          break; // Update successful
        }
        // Update failed, current_max was updated by another thread, retry
      }
#endif
    }
  }

  if (verbose)
    std::cout << "Thread " << thread_id << " finished." << std::endl;
}

/**
 * @brief Worker function for pure cyclic scheduling
 *
 * Each thread processes individual numbers in a round-robin fashion.
 */
void static_cyclic_worker(int thread_id, int num_threads,
                          const std::vector<Range> &global_ranges,
                          std::vector<RangeResult> &results,
                          bool verbose = false) {
  for (size_t range_idx = 0; range_idx < global_ranges.size(); ++range_idx) {
    const auto &current_range = global_ranges[range_idx];

    // Skip invalid ranges where start > end
    if (current_range.start > current_range.end) {
      continue;
    }

    // Track local maximum steps for the current thread
    ull thread_local_max = 0;

    // Process each number in the range that aligns with thread_id
    for (ull num = current_range.start + thread_id; num <= current_range.end;
         num += num_threads) {

      if (verbose && num % 10000 == 0) {
        std::cout << "Thread " << thread_id << " processing number " << num
                  << " in range " << range_idx << std::endl;
      }

      // Compute steps for this number
      ull steps = collatz_steps(num);

      // Update thread-local maximum steps
      thread_local_max = std::max(thread_local_max, steps);
    }

    // Atomically update the maximum steps for this range
    if (thread_local_max > 0) {
#if __cplusplus >= 202002L
      results[range_idx].max_steps.fetch_max(thread_local_max,
                                             std::memory_order_relaxed);
#else
      ull current_max =
          results[range_idx].max_steps.load(std::memory_order_relaxed);
      while (thread_local_max > current_max) {
        if (results[range_idx].max_steps.compare_exchange_weak(
                current_max, thread_local_max, std::memory_order_release,
                std::memory_order_relaxed)) {
          break; // Update successful
        }
        // Update failed, current_max was updated by another thread
      }
#endif
    }
  }

  if (verbose)
    std::cout << "Thread " << thread_id << " finished." << std::endl;
}

/**
 * @brief Executes the computation using the selected static scheduling approach
 *
 * Dispatches to the appropriate worker function based on the static_variant in
 * config.
 *
 * @param config Configuration parameters including thread count, chunk size,
 * static variant and input ranges
 * @param results_out Vector to store the computation results
 * @return bool True if execution was successful, false otherwise
 */
bool run_static_scheduling(const Config &config,
                           std::vector<RangeResult> &results_out) {
  // Validate input parameters
  if (config.num_threads <= 0 ||
      (config.static_variant == StaticVariant::BLOCK_CYCLIC &&
       config.chunk_size == 0))
    return false;

  int optimal_threads =
      std::min(config.num_threads,
               static_cast<int>(std::thread::hardware_concurrency()));

  std::vector<std::thread> threads;
  results_out.clear();

  // Initialize results vector with one entry per input range
  for (const auto &r : config.ranges) {
    results_out.emplace_back(r);
  }

  if (config.verbose) {
    std::string variant;
    switch (config.static_variant) {
    case StaticVariant::BLOCK:
      variant = "Block";
      break;
    case StaticVariant::CYCLIC:
      variant = "Cyclic";
      break;
    case StaticVariant::BLOCK_CYCLIC:
      variant = "Block-Cyclic";
      break;
    }

    std::cout << "Starting static " << variant << " scheduling with "
              << optimal_threads << " threads";

    if (config.static_variant == StaticVariant::BLOCK_CYCLIC) {
      std::cout << " and block size " << config.chunk_size;
    }
    std::cout << std::endl;
  }

  // Create and start worker threads with the appropriate scheduling variant
  switch (config.static_variant) {
  case StaticVariant::BLOCK:
    for (int i = 0; i < optimal_threads; ++i) {
      threads.emplace_back(static_block_worker, i, optimal_threads,
                           std::cref(config.ranges), std::ref(results_out),
                           config.verbose);
    }
    break;

  case StaticVariant::CYCLIC:
    for (int i = 0; i < optimal_threads; ++i) {
      threads.emplace_back(static_cyclic_worker, i, optimal_threads,
                           std::cref(config.ranges), std::ref(results_out),
                           config.verbose);
    }
    break;

  case StaticVariant::BLOCK_CYCLIC:
    for (int i = 0; i < optimal_threads; ++i) {
      threads.emplace_back(static_worker, i, optimal_threads, config.chunk_size,
                           std::cref(config.ranges), std::ref(results_out),
                           config.verbose);
    }
    break;
  }

  // Wait for all threads to complete
  for (auto &t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }

  if (config.verbose) {
    std::cout << "Static scheduling finished." << std::endl;
  }

  return true;
}

/**
 * @brief Legacy function for block-cyclic scheduling (kept for backward
 * compatibility)
 *
 * This implementation simply calls the new unified run_static_scheduling
 * function with the BLOCK_CYCLIC variant.
 */
bool run_static_block_cyclic(const Config &config,
                             std::vector<RangeResult> &results_out) {
  // Create a copy of the config to ensure we use block-cyclic variant
  Config block_cyclic_config = config;
  block_cyclic_config.static_variant = StaticVariant::BLOCK_CYCLIC;

  // Call the new unified implementation
  return run_static_scheduling(block_cyclic_config, results_out);
}
