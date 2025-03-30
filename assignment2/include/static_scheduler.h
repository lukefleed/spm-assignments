#ifndef STATIC_SCHEDULER_H
#define STATIC_SCHEDULER_H

#include "common_types.h"
#include <vector>

/**
 * @brief Executes block-cyclic scheduling.
 *
 * @param config Configuration parameters.
 * @param results_out Vector to store the computation results.
 * @return True if execution was successful, false otherwise.
 */
bool run_static_block_cyclic(const Config &config,
                             std::vector<RangeResult> &results_out);

/**
 * @brief Executes the computation using a static scheduling approach.
 *
 * Supports BLOCK, CYCLIC, and BLOCK_CYCLIC scheduling variants.
 *
 * @param config Configuration parameters including thread count, chunk size,
 *               static variant, and input ranges.
 * @param results_out Vector to store the computation results.
 * @return True if execution was successful, false otherwise.
 */
bool run_static_scheduling(const Config &config,
                           std::vector<RangeResult> &results_out);

#endif // STATIC_SCHEDULER_H
