#ifndef COLLATZ_H
#define COLLATZ_H

#include "common_types.h"

/**
 * @brief Calcola il numero di passi della sequenza di Collatz per n.
 * @param n Il numero di partenza (intero positivo).
 * @return Il numero di passi per raggiungere 1.
 */
ull collatz_steps(ull n);

/**
 * @brief Trova il massimo numero di passi Collatz in un sotto-intervallo
 * [start, end].
 * @param start Inizio dell'intervallo (incluso).
 * @param end Fine dell'intervallo (incluso).
 * @return Il massimo numero di passi trovato nell'intervallo. Ritorna 0 se
 * start > end.
 */
ull find_max_steps_in_subrange(ull start, ull end);

#endif // COLLATZ_H
