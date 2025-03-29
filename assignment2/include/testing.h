#ifndef TESTING_H
#define TESTING_H

#include "common_types.h"
#include <string>
#include <vector>

/**
 * @brief Esegue la suite di test di correttezza.
 * Confronta i risultati delle implementazioni parallele (statica e dinamica)
 * con l'implementazione sequenziale per un set predefinito di casi di test.
 * Stampa i risultati (PASS/FAIL) su std::cout.
 *
 * @return true se tutti i test passano, false altrimenti.
 */
bool run_correctness_suite();

/**
 * @brief Esegue la suite di test di performance.
 * Misura il tempo di esecuzione (mediana su più run) per le implementazioni
 * sequenziale, statica e dinamica su un workload fisso, variando il numero
 * di thread e (opzionalmente) la dimensione del chunk/blocco.
 * Stampa i risultati in formato CSV su std::cout.
 *
 * @param thread_counts Vettore con i numeri di thread da testare.
 * @param chunk_sizes Vettore con le dimensioni di chunk/blocco da testare.
 * @param samples Numero di campioni da raccogliere per ogni configurazione.
 * @param iterations_per_sample Numero di esecuzioni per ogni campione (per
 * ridurre rumore).
 * @param workload Vettore di Range che definisce il carico di lavoro fisso per
 * i test.
 * @return true se l'esecuzione avviene senza errori (non implica correttezza),
 * false altrimenti.
 */
bool run_performance_suite(const std::vector<int> &thread_counts,
                           const std::vector<ull> &chunk_sizes, int samples,
                           int iterations_per_sample,
                           const std::vector<Range> &workload);

#endif // TESTING_H
