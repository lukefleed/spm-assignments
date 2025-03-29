#include "collatz.h"
#include <limits>    // Per std::numeric_limits
#include <stdexcept> // Per std::overflow_error

ull collatz_steps(ull n) {
  if (n == 0) {
    // Il problema è definito per interi *positivi*. Potremmo lanciare eccezione
    // o ritornare 0/valore speciale. Torniamo 0 assumendo che l'input sia
    // sempre >= 1 come da definizione standard.
    return 0; // O lanciare std::invalid_argument("Input n must be positive");
  }
  if (n == 1) {
    return 0; // Già a 1, 0 passi.
  }

  ull steps = 0;
  while (n != 1) {
    if (n % 2 == 0) {
      n /= 2;
    } else {
      // Check per potenziale overflow prima di moltiplicare
      if (n > (std::numeric_limits<ull>::max() - 1) / 3) {
        // Se n è molto grande, 3*n + 1 potrebbe causare overflow.
        // Gestire questo caso è complesso e fuori dallo scope standard del
        // problema Collatz. Potremmo lanciare un'eccezione o stampare un
        // warning. Per ora, assumiamo che non accada con gli input ragionevoli,
        // ma è un limite da tenere presente.
        // throw std::overflow_error("Collatz sequence overflow detected for n =
        // " + std::to_string(n)); Oppure semplicemente procediamo, rischiando
        // l'overflow (comportamento C++ standard)
      }
      n = 3 * n + 1;
    }
    steps++;
    // Aggiungere un controllo potenziale per cicli (anche se la congettura dice
    // che non esistono a parte 1-4-2-1) if (steps > MAX_REASONABLE_STEPS) {
    // throw std::runtime_error("Too many steps"); }
  }
  return steps;
}

ull find_max_steps_in_subrange(ull start, ull end) {
  ull max_s = 0;
  if (start == 0)
    start = 1; // Assicura che partiamo da 1 se l'intervallo include 0
  if (start > end)
    return 0; // Intervallo non valido o vuoto

  for (ull i = start; i <= end; ++i) {
    ull current_steps = collatz_steps(i);
    if (current_steps > max_s) {
      max_s = current_steps;
    }
    // Se 'end' è std::numeric_limits<ull>::max(), l'incremento i++ causerà
    // overflow. Questo è un caso limite improbabile per gli intervalli dati, ma
    // tecnicamente possibile.
    if (i == std::numeric_limits<ull>::max())
      break;
  }
  return max_s;
}
