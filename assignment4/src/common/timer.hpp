#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>
#include <iostream>
#include <string>

/**
 * @brief High-resolution timer for performance measurements
 */
class Timer {
private:
  using clock = std::chrono::high_resolution_clock;
  using time_point = clock::time_point;

  time_point start_time;
  std::string name;
  bool auto_print;

public:
  Timer(const std::string &timer_name = "", bool auto_print_on_destroy = false)
      : name(timer_name), auto_print(auto_print_on_destroy) {
    start();
  }

  ~Timer() {
    if (auto_print) {
      std::cout << name << ": " << elapsed_ms() << " ms\n";
    }
  }

  void start() { start_time = clock::now(); }

  double elapsed_ms() const {
    auto end_time = clock::now();
    return std::chrono::duration<double, std::milli>(end_time - start_time)
        .count();
  }

  double elapsed_us() const {
    auto end_time = clock::now();
    return std::chrono::duration<double, std::micro>(end_time - start_time)
        .count();
  }
};

#endif // TIMER_HPP
