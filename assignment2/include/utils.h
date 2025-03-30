#ifndef UTILS_H
#define UTILS_H

#include "common_types.h"
#include <chrono>
#include <optional>
#include <string>
#include <vector>

/**
 * @brief Parses a string in the format "start-end" into a Range struct.
 * @param s The string to parse.
 * @param range The Range struct to populate.
 * @return True if parsing is successful, false otherwise.
 */
bool parse_range_string(const std::string &s, Range &range);

/**
 * @brief Parses command-line arguments.
 * @param argc Number of arguments.
 * @param argv Array of string arguments.
 * @return A std::optional<Config> containing the configuration if parsing is
 * successful, std::nullopt otherwise (or if help is requested).
 */
std::optional<Config> parse_arguments(int argc, char *argv[]);

/**
 * @brief A simple timer class for measuring elapsed time.
 */
class Timer {
public:
  Timer() : start_time(std::chrono::high_resolution_clock::now()) {}

  /**
   * @brief Resets the timer to the current time.
   */
  void reset() { start_time = std::chrono::high_resolution_clock::now(); }

  /**
   * @brief Returns the elapsed time in milliseconds.
   * @return The elapsed time in milliseconds as a double.
   */
  double elapsed_ms() const {
    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end_time - start_time)
        .count();
  }

  /**
   * @brief Returns the elapsed time in seconds.
   * @return The elapsed time in seconds as a double.
   */
  double elapsed_s() const {
    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end_time - start_time).count();
  }

private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
};

#endif // UTILS_H
