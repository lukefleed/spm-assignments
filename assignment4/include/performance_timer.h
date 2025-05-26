#ifndef PERFORMANCE_TIMER_H
#define PERFORMANCE_TIMER_H

#include <chrono> // For high_resolution_clock, duration

// A simple timer class for measuring execution time of code blocks.
// It uses std::chrono::high_resolution_clock for the most precise timing
// available.
class PerformanceTimer {
public:
  // Constructor: Optionally starts the timer immediately upon creation.
  PerformanceTimer(bool start_immediately = false);

  // Starts or restarts the timer.
  void start();

  // Stops the timer.
  void stop();

  // Returns the elapsed time in seconds as a double.
  // If the timer is still running, it returns the time elapsed up to the call.
  // If stop() has not been called after start(), it effectively measures
  // current duration.
  double elapsed_seconds();

  // Returns the elapsed time in milliseconds as a double.
  double elapsed_milliseconds();

  // Returns the elapsed time in microseconds as a double.
  double elapsed_microseconds();

private:
  // Type alias for high-resolution clock's time point.
  using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

  // Private helper function to get the correct end time point for duration
  // calculation.
  TimePoint get_current_or_end_time() const;

  TimePoint start_time; // Time point when the timer was started.
  TimePoint end_time;   // Time point when the timer was stopped.
  bool running;         // Flag to indicate if the timer is currently running.
};

#endif // PERFORMANCE_TIMER_H
