/**
 * @file file_handler.cpp
 * @brief Implements file discovery and filtering logic.
 */
#include "file_handler.hpp"
#include "config.hpp"
#include <iostream>
#include <optional>

namespace FileHandler {

bool is_directory(const std::filesystem::path &p, int verbosity) {
  std::error_code ec;
  bool is_dir = std::filesystem::is_directory(p, ec);
  if (ec && verbosity >= 1) {
    std::cerr << "Warning: Error checking if path is directory: " << p.string()
              << " - " << ec.message() << std::endl;
    return false;
  }
  return is_dir;
}

std::optional<size_t> get_regular_file_size(const std::filesystem::path &p,
                                            int verbosity) {
  std::error_code ec;
  if (!std::filesystem::is_regular_file(p, ec)) {
    if (ec && verbosity >= 1) {
      std::cerr << "Warning: Error checking if path is regular file: "
                << p.string() << " - " << ec.message() << std::endl;
    }
    return std::nullopt;
  }
  std::error_code size_ec;
  uintmax_t size = std::filesystem::file_size(p, size_ec);
  if (size_ec) {
    if (verbosity >= 1) {
      std::cerr << "Warning: Could not get file size for: " << p.string()
                << " - " << size_ec.message() << std::endl;
    }
    return std::nullopt;
  }
  return static_cast<size_t>(size);
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
    return !has_suffix;
  } else {
    return has_suffix;
  }
}

std::vector<WorkItem>
discover_work_items(const std::vector<std::string> &initial_paths,
                    const ConfigData &cfg) {
  std::vector<WorkItem> items_to_process;
  std::vector<std::filesystem::path> directories_to_scan;

  for (const auto &path_str : initial_paths) {
    std::filesystem::path current_path(path_str);
    if (is_directory(current_path, cfg.verbosity)) {
      directories_to_scan.push_back(current_path);
    } else {
      auto filesize_opt = get_regular_file_size(current_path, cfg.verbosity);
      if (filesize_opt) {
        if (should_process(current_path.filename().string(), cfg.compress_mode,
                           SUFFIX)) {
          items_to_process.push_back(
              {current_path.string(), filesize_opt.value()});
        } else if (cfg.verbosity >= 2) {
          std::cout << "Skipping initial path (suffix mismatch): "
                    << current_path.string() << std::endl;
        }
      } else {
        std::error_code exist_ec;
        if (!std::filesystem::exists(current_path, exist_ec)) {
          if (cfg.verbosity >= 1) {
            std::cerr << "Warning: Initial path not found or inaccessible: "
                      << path_str << std::endl;
          }
        } else if (cfg.verbosity >= 1) {
          std::cerr << "Warning: Skipping initial path (not a processable "
                       "regular file): "
                    << path_str << std::endl;
        }
      }
    }
  }

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
      continue;
    }
    for (const auto &entry : dir_iter) {
      try {
        if (is_directory(entry.path(), cfg.verbosity)) {
          if (cfg.recurse) {
            directories_to_scan.push_back(entry.path());
          }
        } else {
          auto filesize_opt =
              get_regular_file_size(entry.path(), cfg.verbosity);
          if (filesize_opt) {
            if (should_process(entry.path().filename().string(),
                               cfg.compress_mode, SUFFIX)) {
              items_to_process.push_back(
                  {entry.path().string(), filesize_opt.value()});
            } else if (cfg.verbosity >= 2) {
              std::cout << "Skipping file (suffix mismatch): "
                        << entry.path().string() << std::endl;
            }
          }
        }
      } catch (const std::filesystem::filesystem_error &e) {
        if (cfg.verbosity >= 1) {
          std::cerr << "Warning: Filesystem error processing entry near "
                    << e.path1().string() << ": " << e.what() << std::endl;
        }
        continue;
      }
    }
  }
  return items_to_process;
}

} // namespace FileHandler
