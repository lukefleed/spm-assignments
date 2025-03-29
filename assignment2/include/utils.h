#ifndef UTILS_H
#define UTILS_H

#include "common_types.h"
#include <chrono>
#include <optional> // For std::optional
#include <string>
#include <vector>

/**
 * @brief Effettua il parsing di una stringa "start-end" in una struct Range.
 * @param s La stringa da parsare.
 * @param range La struct Range da popolare.
 * @return true se il parsing ha successo, false altrimenti.
 */
bool parse_range_string(const std::string &s, Range &range);

/**
 * @brief Effettua il parsing degli argomenti della linea di comando.
 * @param argc Numero di argomenti.
 * @param argv Array di argomenti stringa.
 * @return Un std::optional<Config> contenente la configurazione se il parsing
 * ha successo, std::nullopt altrimenti (o se viene richiesto aiuto).
 */
std::optional<Config> parse_arguments(int argc, char *argv[]);

// Semplice classe Timer
class Timer {
public:
  Timer() : start_time(std::chrono::high_resolution_clock::now()) {}

  void reset() { start_time = std::chrono::high_resolution_clock::now(); }

  // Ritorna il tempo trascorso in millisecondi
  double elapsed_ms() const {
    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end_time - start_time)
        .count();
  }

  // Ritorna il tempo trascorso in secondi
  double elapsed_s() const {
    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end_time - start_time).count();
  }

private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
};

#endif // UTILS_H
