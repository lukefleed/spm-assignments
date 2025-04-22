#include "file_handler.hpp"
#include <iostream> // For cerr in case of errors

namespace FileHandler {

bool is_directory_or_file(const std::filesystem::path &p, size_t &filesize,
                          int verbosity) {
  std::error_code ec;
  if (std::filesystem::is_directory(p, ec)) {
    filesize = 0; // Directories don't have a relevant size here
    return true;
  }
  if (ec && verbosity >= 1) { // Handle potential error from is_directory check
    std::cerr << "Warning: Error checking if path is directory: " << p.string()
              << " - " << ec.message() << std::endl;
  }
  ec.clear(); // Reset error code

  if (std::filesystem::is_regular_file(p, ec)) {
    std::error_code size_ec;
    filesize = std::filesystem::file_size(p, size_ec);
    if (size_ec && verbosity >= 1) {
      std::cerr << "Warning: Could not get file size for: " << p.string()
                << " - " << size_ec.message() << std::endl;
      filesize = 0; // Indicate error or unknown size
    }
    return false; // It's a file
  }
  if (ec &&
      verbosity >= 1) { // Handle potential error from is_regular_file check
    std::cerr << "Warning: Error checking if path is regular file: "
              << p.string() << " - " << ec.message() << std::endl;
  }

  // If neither directory nor regular file (or error occurred)
  filesize = 0;
  if (verbosity >= 1 &&
      std::filesystem::exists(p)) { // Only warn if it exists but isn't dir/file
    std::cerr << "Warning: Skipping path (not a regular file or directory): "
              << p.string() << std::endl;
  }
  // Treat as non-file for processing purposes, maybe throw an exception if
  // stricter handling is needed
  return false;
}

bool should_process(const std::string &filename, bool is_compress_mode,
                    const std::string &suffix) {
  if (filename == "." || filename == "..") {
    return false;
  }
  bool has_suffix = (filename.length() >= suffix.length()) &&
                    (filename.compare(filename.length() - suffix.length(),
                                      suffix.length(), suffix) == 0);

  if (is_compress_mode) {
    return !has_suffix; // Compress if it doesn't have the suffix
  } else {
    return has_suffix; // Decompress if it has the suffix
  }
}

std::vector<WorkItem>
discover_work_items(const std::vector<std::string> &initial_paths,
                    const ConfigData &cfg) {
  std::vector<WorkItem> items_to_process;
  std::vector<std::filesystem::path> directories_to_scan;

  // Initial population from command line arguments
  for (const auto &path_str : initial_paths) {
    std::filesystem::path current_path(path_str);
    size_t filesize = 0;
    bool is_dir = is_directory_or_file(current_path, filesize, cfg.verbosity);

    if (is_dir) {
      directories_to_scan.push_back(current_path);
    } else if (filesize > 0 ||
               std::filesystem::exists(
                   current_path)) { // Process if it's a file (even 0-byte) or
                                    // if is_directory_or_file had error but
                                    // path exists
      if (should_process(current_path.filename().string(), cfg.compress_mode,
                         SUFFIX)) {
        items_to_process.push_back({current_path.string(), filesize, false});
      } else if (cfg.verbosity >= 2) {
        std::cout << "Skipping initial path (suffix mismatch): "
                  << current_path.string() << std::endl;
      }
    } else if (cfg.verbosity >= 1) {
      std::cerr << "Warning: Initial path not found or inaccessible: "
                << path_str << std::endl;
    }
  }

  // Process directories found initially and potentially recursively
  size_t current_dir_index = 0;
  while (current_dir_index < directories_to_scan.size()) {
    const auto &dir_path = directories_to_scan[current_dir_index++];
    if (cfg.verbosity >= 2) {
      std::cout << "Scanning directory: " << dir_path.string() << std::endl;
    }

    std::error_code ec;
    auto iterator_options =
        std::filesystem::directory_options::skip_permission_denied;
    std::filesystem::directory_iterator dir_iter(dir_path, iterator_options,
                                                 ec);

    if (ec && cfg.verbosity >= 1) {
      std::cerr << "Warning: Cannot open directory: " << dir_path.string()
                << " - " << ec.message() << std::endl;
      continue; // Skip this directory
    }

    for (const auto &entry : dir_iter) {
      try {
        size_t filesize = 0;
        bool is_dir =
            is_directory_or_file(entry.path(), filesize, cfg.verbosity);

        if (is_dir) {
          if (cfg.recurse) {
            directories_to_scan.push_back(
                entry.path()); // Add to scan list if recursing
          }
        } else { // It's a file
          if (should_process(entry.path().filename().string(),
                             cfg.compress_mode, SUFFIX)) {
            items_to_process.push_back(
                {entry.path().string(), filesize, false});
          } else if (cfg.verbosity >= 2) {
            std::cout << "Skipping file (suffix mismatch): "
                      << entry.path().string() << std::endl;
          }
        }
      } catch (const std::filesystem::filesystem_error &e) {
        if (cfg.verbosity >= 1) {
          std::cerr << "Warning: Filesystem error processing entry "
                    << e.path1().string() << ": " << e.what() << std::endl;
        }
        continue; // Skip this entry on error
      }
    }
  }

  return items_to_process;
}

} // namespace FileHandler
