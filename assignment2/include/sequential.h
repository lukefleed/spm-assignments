#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include "common_types.h"
#include <vector>

/**
 * @brief Esegue il calcolo del massimo numero di passi Collatz sequenzialmente
 * per ogni range.
 * @param ranges Vettore dei range di input.
 * @return Vettore contenente il massimo numero di passi per ogni range
 * corrispondente.
 */
std::vector<ull> run_sequential(const std::vector<Range> &ranges);

#endif // SEQUENTIAL_H
