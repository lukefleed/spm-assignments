#ifndef COMMON_TYPES_H
#define COMMON_TYPES_H

#include <atomic>
#include <optional> // Per C++17
#include <string>
#include <vector>

// Alias per unsigned long long per chiarezza
using ull = unsigned long long;

// Struttura per rappresentare un intervallo
struct Range {
  ull start;
  ull end;
};

// Struttura per rappresentare un task (usato nello scheduler dinamico)
struct Task {
  ull start;
  ull end;
  size_t original_range_index; // Indice del range originale da cui proviene il
                               // task
};

// Struttura per memorizzare il risultato (massimo trovato) per un range,
// usando std::atomic per l'aggiornamento thread-safe.
struct RangeResult {
  Range original_range;
  std::atomic<ull> max_steps{0}; // Inizializzato a 0

  // Costruttore per inizializzare il range originale
  RangeResult(const Range &r) : original_range(r) {}
  // Serve un costruttore di copia/move o default se usi vector<RangeResult>
  RangeResult(const RangeResult &other)
      : original_range(other.original_range),
        max_steps(other.max_steps.load()) {}
  RangeResult &operator=(const RangeResult &other) {
    if (this != &other) {
      original_range = other.original_range;
      max_steps.store(other.max_steps.load());
    }
    return *this;
  }
  // Necessario per std::vector::emplace_back se non si definisce costruttore di
  // copia/move
  RangeResult() = default;
};

// Struttura per contenere la configurazione del programma letta dagli argomenti
enum class SchedulingType { SEQUENTIAL, STATIC, DYNAMIC };

// Nuovo enum per varianti di scheduling statico
enum class StaticVariant { BLOCK, CYCLIC, BLOCK_CYCLIC };

struct Config {
  SchedulingType scheduling = SchedulingType::STATIC; // Default statico
  StaticVariant static_variant =
      StaticVariant::BLOCK_CYCLIC; // Default block-cyclic
  int num_threads = 16;            // Default threads
  ull chunk_size = 1;              // Default chunk/block size
  std::vector<Range> ranges;
  bool verbose = false; // Opzionale per debug
};

#endif // COMMON_TYPES_H
