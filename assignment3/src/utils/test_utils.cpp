#include "test_utils.hpp"

#include <cstring>    // For strerror
#include <filesystem> // For directory cleanup
#include <fstream>
#include <iostream> // For error messages
#include <random>   // For fallback random generation
#include <vector>

namespace TestUtils {

// Platform-specific random data source (like /dev/urandom)
#if defined(__linux__) || defined(__APPLE__)
#include <fcntl.h>
#include <unistd.h>
#define USE_DEV_URANDOM 1
#else
#define USE_DEV_URANDOM 0
#endif

bool create_random_file(const std::string &path, size_t size, int verbosity) {
  std::ofstream out_file(path, std::ios::binary | std::ios::trunc);
  if (!out_file) {
    if (verbosity >= 1)
      std::cerr << "Error: Cannot create file for testing: " << path
                << std::endl;
    return false;
  }
  if (size == 0) { // Handle 0-byte case
    out_file.close();
    return true;
  }

  // Declare buffer and bytes_written outside the conditional block
  std::vector<char> buffer(std::min((size_t)4096, size));
  size_t bytes_written = 0;

#if USE_DEV_URANDOM
  int rand_fd = open("/dev/urandom", O_RDONLY);
  if (rand_fd < 0) {
    if (verbosity >= 1)
      std::cerr << "Warning: Cannot open /dev/urandom, falling back to "
                   "pseudo-random data."
                << std::endl;
    goto fallback_random; // Use goto for fallback logic here
  }

  // Use the already declared buffer and bytes_written
  while (bytes_written < size) {
    size_t bytes_to_read = std::min(buffer.size(), size - bytes_written);
    ssize_t bytes_read = read(rand_fd, buffer.data(), bytes_to_read);
    if (bytes_read <= 0) {
      if (verbosity >= 1)
        std::cerr << "Error reading from /dev/urandom." << std::endl;
      close(rand_fd);
      out_file.close();
      unlink(path.c_str()); // Clean up partial file
      return false;
    }
    out_file.write(buffer.data(), bytes_read);
    if (!out_file) {
      if (verbosity >= 1)
        std::cerr << "Error writing random data to file: " << path << std::endl;
      close(rand_fd);
      out_file.close();
      unlink(path.c_str());
      return false;
    }
    bytes_written += bytes_read;
  }
  close(rand_fd);
  out_file.close();
  return true;

fallback_random: // Label for fallback mechanism
  // Reset bytes_written for the fallback mechanism
  bytes_written = 0;
#endif // USE_DEV_URANDOM

  // Fallback using C++ <random> - less ideal for binary data but works
  std::mt19937 gen(std::random_device{}()); // Standard mersenne_twister_engine
                                            // seeded with random_device
  std::uniform_int_distribution<char> distrib; // Distribution for chars
  // Use the already declared buffer and bytes_written
  while (bytes_written < size) {
    size_t bytes_to_generate = std::min(buffer.size(), size - bytes_written);
    for (size_t i = 0; i < bytes_to_generate; ++i) {
      buffer[i] = distrib(gen);
    }
    out_file.write(buffer.data(), bytes_to_generate);
    if (!out_file) {
      if (verbosity >= 1)
        std::cerr << "Error writing pseudo-random data to file: " << path
                  << std::endl;
      out_file.close();
      unlink(path.c_str());
      return false;
    }
    bytes_written += bytes_to_generate;
  }

  out_file.close();
  return true;
}

bool compare_files(const std::string &path1, const std::string &path2,
                   int verbosity) {
  std::ifstream file1(path1,
                      std::ios::binary | std::ios::ate); // ate = open at end
  std::ifstream file2(path2, std::ios::binary | std::ios::ate);

  if (!file1) {
    if (verbosity >= 1)
      std::cerr << "Error: Cannot open file for comparison: " << path1
                << std::endl;
    return false;
  }
  if (!file2) {
    if (verbosity >= 1)
      std::cerr << "Error: Cannot open file for comparison: " << path2
                << std::endl;
    return false;
  }

  std::ifstream::pos_type size1 = file1.tellg();
  std::ifstream::pos_type size2 = file2.tellg();

  if (size1 != size2) {
    if (verbosity >= 1)
      std::cerr << "Files differ in size: " << path1 << " (" << size1 << ") vs "
                << path2 << " (" << size2 << ")" << std::endl;
    return false; // Files differ if sizes are different
  }

  // Reset pointers to beginning
  file1.seekg(0, std::ios::beg);
  file2.seekg(0, std::ios::beg);

  // Compare content block by block
  const size_t buffer_size = 4096;
  std::vector<char> buffer1(buffer_size);
  std::vector<char> buffer2(buffer_size);

  while (file1 && file2) {
    file1.read(buffer1.data(), buffer_size);
    file2.read(buffer2.data(), buffer_size);

    // Get actual bytes read (might be less than buffer_size at the end)
    std::streamsize bytes_read1 = file1.gcount();
    std::streamsize bytes_read2 = file2.gcount();

    if (bytes_read1 != bytes_read2) {
      // This should not happen if sizes were equal, but check defensively
      if (verbosity >= 1)
        std::cerr << "Internal comparison error: Read count mismatch."
                  << std::endl;
      return false;
    }
    if (bytes_read1 == 0) {
      break; // End of both files
    }

    if (memcmp(buffer1.data(), buffer2.data(), bytes_read1) != 0) {
      if (verbosity >= 1)
        std::cerr << "Files differ in content: " << path1 << " vs " << path2
                  << std::endl;
      return false; // Content mismatch
    }
  }

  // Check for any stream errors after loop
  if (!file1.eof() && file1.fail()) {
    if (verbosity >= 1)
      std::cerr << "Error reading file during comparison: " << path1
                << std::endl;
    return false;
  }
  if (!file2.eof() && file2.fail()) {
    if (verbosity >= 1)
      std::cerr << "Error reading file during comparison: " << path2
                << std::endl;
    return false;
  }

  return true; // Files are identical
}

bool clean_files_with_suffix(const std::string &directory,
                             const std::string &suffix, bool recursive,
                             int verbosity) {
  bool success = true;
  try {
    std::filesystem::path dir_path(directory);
    if (!std::filesystem::exists(dir_path) ||
        !std::filesystem::is_directory(dir_path)) {
      if (verbosity >= 1)
        std::cerr << "Warning: Directory not found for cleaning: " << directory
                  << std::endl;
      return true; // Not an error if dir doesn't exist
    }

    auto iterator_options =
        std::filesystem::directory_options::skip_permission_denied;
    auto process_entry = [&](const std::filesystem::directory_entry &entry) {
      try {
        if (entry.is_regular_file()) {
          if (entry.path().extension() == suffix) {
            std::error_code ec;
            std::filesystem::remove(entry.path(), ec);
            if (ec) {
              if (verbosity >= 1)
                std::cerr << "Warning: Failed to remove file: "
                          << entry.path().string() << " - " << ec.message()
                          << std::endl;
              success = false;
            } else if (verbosity >= 2) {
              std::cout << "Removed: " << entry.path().string() << std::endl;
            }
          }
        }
      } catch (const std::filesystem::filesystem_error &e) {
        if (verbosity >= 1)
          std::cerr << "Warning: Filesystem error processing entry "
                    << e.path1().string() << ": " << e.what() << std::endl;
        success = false;
      }
    };

    if (recursive) {
      for (const auto &entry : std::filesystem::recursive_directory_iterator(
               dir_path, iterator_options)) {
        process_entry(entry);
      }
    } else {
      for (const auto &entry :
           std::filesystem::directory_iterator(dir_path, iterator_options)) {
        process_entry(entry);
      }
    }

  } catch (const std::filesystem::filesystem_error &e) {
    if (verbosity >= 1)
      std::cerr << "Error during directory cleaning: " << e.what() << std::endl;
    return false;
  }
  return success;
}

} // namespace TestUtils
