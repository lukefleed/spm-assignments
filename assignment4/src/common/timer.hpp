#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>
#include <iostream>
#include <string>

/**
 * @brief High-resolution timer for performance measurements
 *
 * Uses std::chrono::high_resolution_clock for maximum precision.
 * Not thread-safe - each thread should use separate Timer instances.
 */
class Timer {
private:
  using clock = std::chrono::high_resolution_clock;
  using time_point = clock::time_point;

  time_point start_time;
  std::string name;
  bool auto_print;

public:
  /**
   * @brief Constructs timer and immediately starts measurement
   * @param timer_name Optional name for identification in output
   * @param auto_print_on_destroy If true, prints elapsed time in destructor
   */
  Timer(const std::string &timer_name = "", bool auto_print_on_destroy = false)
      : name(timer_name), auto_print(auto_print_on_destroy) {
    start();
  }

  /**
   * @brief Destructor with optional automatic timing output
   */
  ~Timer() {
    if (auto_print) {
      std::cout << name << ": " << elapsed_ms() << " ms\n";
    }
  }

  /**
   * @brief Resets timer to current time point
   */
  void start() { start_time = clock::now(); }

  /**
   * @brief Returns elapsed time in milliseconds since last start()
   * @return Elapsed time as double precision milliseconds
   */
  double elapsed_ms() const {
    auto end_time = clock::now();
    return std::chrono::duration<double, std::milli>(end_time - start_time)
        .count();
  }

  /**
   * @brief Returns elapsed time in microseconds since last start()
   * @return Elapsed time as double precision microseconds
   */
  double elapsed_us() const {
    auto end_time = clock::now();
    return std::chrono::duration<double, std::micro>(end_time - start_time)
        .count();
  }
};

#endif // TIMER_HPP
