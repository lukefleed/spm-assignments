#include "performance_timer.h"

// Constructor implementation.
PerformanceTimer::PerformanceTimer(bool start_immediately) : running(false) {
  if (start_immediately) {
    start();
  }
}

// Starts or restarts the timer.
void PerformanceTimer::start() {
  start_time = std::chrono::high_resolution_clock::now();
  running = true;
}

// Stops the timer.
void PerformanceTimer::stop() {
  end_time = std::chrono::high_resolution_clock::now();
  running = false;
}

// Calculates elapsed time. Internal helper to avoid code duplication.
PerformanceTimer::TimePoint PerformanceTimer::get_current_or_end_time() const {
  if (running) {
    return std::chrono::high_resolution_clock::now();
  }
  return end_time;
}

// Returns the elapsed time in seconds.
double PerformanceTimer::elapsed_seconds() {
  std::chrono::duration<double> diff = get_current_or_end_time() - start_time;
  return diff.count();
}

// Returns the elapsed time in milliseconds.
double PerformanceTimer::elapsed_milliseconds() {
  std::chrono::duration<double, std::milli> diff =
      get_current_or_end_time() - start_time;
  return diff.count();
}

// Returns the elapsed time in microseconds.
double PerformanceTimer::elapsed_microseconds() {
  std::chrono::duration<double, std::micro> diff =
      get_current_or_end_time() - start_time;
  return diff.count();
}
