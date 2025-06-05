#include "src/common/utils.hpp"
#include <iostream>

// Copiamo solo la dichiarazione della funzione
void ff_pipeline_two_farms_mergesort(std::vector<Record> &data,
                                     size_t num_threads);

int main() {
  std::cout << "=== Debug FastFlow Two Farms ===" << std::endl;

  // Generiamo dati di test piccoli
  auto data = generate_data_vector(10, 8, DataPattern::RANDOM);

  std::cout << "\nDati originali:" << std::endl;
  for (size_t i = 0; i < data.size(); ++i) {
    std::cout << "Index " << i << ": key=" << data[i].key << std::endl;
  }

  auto data_copy = copy_records_vector(data);

  std::cout << "\nDati copiati prima del sort:" << std::endl;
  for (size_t i = 0; i < data_copy.size(); ++i) {
    std::cout << "Index " << i << ": key=" << data_copy[i].key << std::endl;
  }

  // Eseguiamo il sort
  ff_pipeline_two_farms_mergesort(data_copy, 4);

  std::cout << "\nDati dopo il sort:" << std::endl;
  for (size_t i = 0; i < data_copy.size(); ++i) {
    std::cout << "Index " << i << ": key=" << data_copy[i].key << std::endl;
  }

  bool sorted = is_sorted_vector(data_copy);
  std::cout << "\nE' ordinato? " << (sorted ? "SI" : "NO") << std::endl;

  return 0;
}
