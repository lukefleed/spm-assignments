#ifndef STATIC_SCHEDULER_H
#define STATIC_SCHEDULER_H

#include "common_types.h"
#include <vector>

/**
 * @brief Esegue il calcolo usando scheduling statico block-cyclic.
 * @param ranges Vettore dei range di input.
 * @param num_threads Numero di thread da usare.
 * @param block_size Dimensione del blocco per l'assegnazione ciclica.
 * @param results_out Vettore (inizializzato) dove salvare i risultati
 * (RangeResult).
 * @return true se successo, false altrimenti.
 */
bool run_static_block_cyclic(const Config &config,
                             std::vector<RangeResult> &results_out);

#endif // STATIC_SCHEDULER_H
