#ifndef STATIC_SCHEDULER_H
#define STATIC_SCHEDULER_H

#include "common_types.h" // Includes Config, RangeResult, StaticVariant, ull
#include <vector>         // For std::vector<RangeResult>

/**
 * @brief Executes the Collatz computation using a specific static scheduling
 * strategy.
 *
 * This function orchestrates the parallel execution based on the
 * `static_variant` specified in the `config`. It divides the work defined by
 * `config.ranges` among `config.num_threads` worker threads according to the
 * chosen variant (BLOCK, CYCLIC, or BLOCK_CYCLIC) and collects the results in
 * `results_out`.
 *
 * @param config The configuration object containing all parameters for the run,
 * including:
 *               - `num_threads`: The number of worker threads to use.
 *               - `static_variant`: The specific static scheduling strategy
 * (BLOCK, CYCLIC, BLOCK_CYCLIC).
 *               - `chunk_size`: The block size (used only for BLOCK_CYCLIC
 * variant).
 *               - `ranges`: The vector of input ranges to process.
 *               - `verbose`: Flag for enabling diagnostic output.
 * @param[out] results_out A vector that will be populated with `RangeResult`
 * objects, one for each input range, containing the maximum steps found. Any
 * existing content in this vector will be cleared.
 * @return True if the execution setup and thread management complete
 * successfully, false otherwise (e.g., due to invalid configuration
 * parameters).
 */
bool run_static_scheduling(const Config &config,
                           std::vector<RangeResult> &results_out);

/**
 * @brief Executes static block-cyclic scheduling. [DEPRECATED]
 *
 * This function is provided for potential backward compatibility. It configures
 * the static variant to BLOCK_CYCLIC and then calls the main
 * `run_static_scheduling` function.
 *
 * @param config Configuration parameters. The `static_variant` field will be
 * overridden.
 * @param results_out Vector to store the computation results.
 * @return True if execution was successful, false otherwise.
 * @deprecated Prefer calling `run_static_scheduling` directly with
 * `config.static_variant` set to `StaticVariant::BLOCK_CYCLIC`.
 */
bool run_static_block_cyclic(const Config &config,
                             std::vector<RangeResult> &results_out);

#endif // STATIC_SCHEDULER_H
