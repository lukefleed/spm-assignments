#ifndef STATIC_SCHEDULER_H
#define STATIC_SCHEDULER_H

#include "common_types.h"
#include <vector>

/**
 * @brief Legacy function for block-cyclic scheduling (kept for backward
 * compatibility)
 */
bool run_static_block_cyclic(const Config &config,
                             std::vector<RangeResult> &results_out);

/**
 * @brief Executes the computation using the selected static scheduling approach
 *
 * Supports BLOCK, CYCLIC, and BLOCK_CYCLIC variants
 *
 * @param config Configuration parameters including thread count, chunk size,
 *               static variant and input ranges
 * @param results_out Vector to store the computation results
 * @return bool True if execution was successful, false otherwise
 */
bool run_static_scheduling(const Config &config,
                           std::vector<RangeResult> &results_out);

#endif // STATIC_SCHEDULER_H
