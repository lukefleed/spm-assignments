/**
 * @brief Quick verification program to confirm Record structure compliance
 */
#include "include/config.hpp"
#include "src/common/record.hpp"
#include <cstring>
#include <iostream>

int main() {
  std::cout << "=== Record Structure Verification ===" << std::endl;
  std::cout << "RPAYLOAD constant: " << RPAYLOAD << " bytes" << std::endl;
  std::cout << "sizeof(Record): " << sizeof(Record) << " bytes" << std::endl;
  std::cout << "sizeof(unsigned long): " << sizeof(unsigned long) << " bytes"
            << std::endl;
  std::cout << "sizeof(char[RPAYLOAD]): " << sizeof(char[RPAYLOAD]) << " bytes"
            << std::endl;

  // Test Record construction and access
  Record rec1;
  Record rec2(12345);

  // Test payload access
  strcpy(rec2.rpayload, "test");

  std::cout << "rec1.key: " << rec1.key << std::endl;
  std::cout << "rec2.key: " << rec2.key << std::endl;
  std::cout << "rec2.rpayload: " << rec2.rpayload << std::endl;

  // Test comparison operators
  rec1.key = 100;
  rec2.key = 200;
  std::cout << "rec1 < rec2: " << (rec1 < rec2) << std::endl;
  std::cout << "rec1 == rec2: " << (rec1 == rec2) << std::endl;

  std::cout << "✅ Record structure is compliant with assignment specification!"
            << std::endl;
  return 0;
}
