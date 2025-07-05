/**
 * @file compressor.cpp
 * @brief Implements the core compression and decompression logic using Miniz
 * and OpenMP for block-based large files.
 */

#include "compressor.hpp"
#include "config.hpp"
#include "miniz.h"

#include <atomic>
#include <cstring> // For strerror
#include <fcntl.h> // For open
#include <filesystem>
#include <fstream>
#include <iostream>     // For errors/verbose output
#include <omp.h>        // For parallel block compression
#include <sys/mman.h>   // For mmap/munmap
#include <sys/stat.h>   // For file stats, fstat
#include <system_error> // For error codes
#include <unistd.h>     // For close, ftruncate, unlink
#include <vector>

namespace Compressor {
using ::FORMAT_VERSION;
using ::MAGIC_NUMBER_LARGE_FILE;
using ::SUFFIX;

//-----------------------------------------------------------------------------
// Internal Helper Functions (Anonymous Namespace)
//-----------------------------------------------------------------------------
namespace { // Start anonymous namespace

// --- Memory Mapping Utilities ---

/**
 * @class MappedFile
 * @brief RAII wrapper for memory-mapped file pointers using POSIX mmap.
 *        Manages the mapped memory region and the associated file descriptor.
 */
class MappedFile {
  unsigned char *ptr_ = nullptr; //< Pointer to the mapped memory region
  size_t size_ = 0;              //< Size of the mapped region in bytes
  int fd_ = -1;                  //< File descriptor that is managed internally

public:
  MappedFile() = default;
  // Delete copy constructor and assignment operator
  MappedFile(const MappedFile &) = delete;
  MappedFile &operator=(const MappedFile &) = delete;
  // Move constructor
  MappedFile(MappedFile &&other) noexcept
      : ptr_(other.ptr_), size_(other.size_), fd_(other.fd_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
    other.fd_ = -1;
  }
  // Move assignment operator
  MappedFile &operator=(MappedFile &&other) noexcept {
    if (this != &other) {
      unmap(); // Unmap existing if any
      ptr_ = other.ptr_;
      size_ = other.size_;
      fd_ = other.fd_;
      other.ptr_ = nullptr;
      other.size_ = 0;
      other.fd_ = -1;
    }
    return *this;
  }

  // This is the key of the pattern RAII: it calls unmap() automatically
  ~MappedFile() noexcept { unmap(); }

  /**
   * @brief Maps a file into memory using mmap system call.
   *
   * This function opens a file and creates a memory mapping for it. It handles
   * file size detection automatically if size_in_out is 0, and properly manages
   * file descriptors based on the mapping type.
   *
   * @param fname The path to the file to be mapped
   * @param size_in_out [in/out] Size of the mapping. If 0 on input, the file
   * size will be determined automatically and returned in this parameter
   * @param prot Memory protection flags for mmap (PROT_READ, PROT_WRITE, etc.)
   * @param flags Memory mapping flags (MAP_PRIVATE, MAP_SHARED, etc.)
   * @param open_flags File opening flags (default: O_RDONLY)
   *
   * @return true if mapping was successful, false otherwise
   *
   * @note This function automatically calls unmap() to release any previous
   * mapping
   * @note For empty files, no actual mapping is created and ptr_ is set to
   * nullptr
   * @note File descriptor management varies based on protection and mapping
   * flags:
   *       - For read-only or private mappings, fd is closed after mapping
   *       - For shared writable mappings, fd may be kept open for potential
   * fsync operations
   * @note On failure, all resources are properly cleaned up and member
   * variables reset
   */
  bool map(const char *fname, size_t &size_in_out, int prot, int flags,
           int open_flags = O_RDONLY) {
    unmap(); // Ensure previous mapping is released

    // Open the file with the specified flags
    fd_ = open(fname, open_flags);
    if (fd_ < 0) {
      std::cerr << "Error: Failed opening file " << fname << " - "
                << strerror(errno) << std::endl;
      return false;
    }

    // If size_in_out is 0, determine the file size
    if (size_in_out == 0) {
      struct stat s; // Use stat to get file size
      if (fstat(fd_, &s) != 0) {
        std::cerr << "Error: Failed to fstat file " << fname << " - "
                  << strerror(errno) << std::endl;
        close(fd_);
        fd_ = -1;
        return false;
      }
      size_in_out = s.st_size; // Update size_in_out with the file size
      if (size_in_out == 0) {  // Handle empty file case
        close(fd_);
        fd_ = -1;
        size_ = 0;
        ptr_ = nullptr; // No mapping needed for empty file
        return true;
      }
    }

    // Memory map the file with the specified size and flags
    ptr_ = static_cast<unsigned char *>(
        mmap(nullptr, size_in_out, prot, flags, fd_, 0));
    if (ptr_ == MAP_FAILED) {
      std::cerr << "Error: Failed to memory map file " << fname << " - "
                << strerror(errno) << std::endl;
      ptr_ = nullptr; // Reset pointer on failure
      close(fd_);
      fd_ = -1;
      return false;
    }
    size_ = size_in_out; // Store the size of the mapped region
    // Keep fd_ open for mapped read-only files, close for others later if
    // needed

    // For writeable mappings (MAP_SHARED), we typically close fd after mapping.
    // For read-only (MAP_PRIVATE), keeping fd open can be valid. Let's close it
    // for simplicity unless needed.
    if (!(prot & PROT_WRITE) || (flags & MAP_PRIVATE)) {
      // If read-only or private, we can close fd after mapping.
      // If MAP_SHARED and PROT_WRITE, keep fd open if needed for later fsync,
      // etc. Assume we don't need fd after mapping for now.
      close(fd_);
      fd_ = -1;
    }

    return true;
  }

  /**
   * @brief Allocates and memory-maps a file for read/write operations.
   *
   * This function creates or truncates a file to the specified size and maps it
   * into memory for efficient access. It handles special cases like zero-size
   * files and provides proper error handling with cleanup on failure.
   *
   * @param fname The path to the file to create/open and map
   * @param size The size in bytes to allocate for the file. If 0, only
   * creates/truncates the file without mapping
   * @param prot Memory protection flags for the mapping (default: PROT_READ |
   * PROT_WRITE)
   * @param flags Memory mapping flags (default: MAP_SHARED)
   *
   * @return true if the operation succeeds, false otherwise
   *
   * @note For zero-size files, the function creates/truncates the file but
   * doesn't perform memory mapping
   * @note The file descriptor remains open for shared writable mappings
   * @note On failure, all allocated resources are properly cleaned up
   * @note Calls unmap() at the beginning to clean up any existing mappings
   */
  bool allocate_and_map(const char *fname, size_t size,
                        int prot = PROT_READ | PROT_WRITE,
                        int flags = MAP_SHARED) {
    unmap();
    size_ = size; // Store intended size

    // Open the file with read/write and create/truncate flags
    fd_ = open(fname, O_RDWR | O_CREAT | O_TRUNC, 0666);
    if (fd_ < 0) {
      std::cerr << "Error: Failed creating/opening output file " << fname
                << " - " << strerror(errno) << std::endl;
      return false;
    }

    if (size > 0) {                   // ftruncate fails for size 0 sometimes
      if (ftruncate(fd_, size) < 0) { // Extends the file without writing
        std::cerr << "Error: ftruncate failed for " << fname << " - "
                  << strerror(errno) << std::endl;

        close(fd_);
        fd_ = -1;
        return false;
      }
    } else {
      // If size is 0, no need to map, just return success after
      // creating/truncating
      close(fd_);
      fd_ = -1;
      ptr_ = nullptr;
      return true;
    }

    // Here ftruncate succeeded, it tells the filesystem that the file will have
    // size "X" but it doesn't actually allocate space. This is called a "sparse
    // file" or "hole". If then I call a memcpy to `ptr_` the OS will allocate
    // the space on demand and use only the disk pages that we actually touch.
    // This is extremely efficient for large files since it avoids allocating
    // space for the entire file upfront and the writing of empty blocks.
    ptr_ =
        static_cast<unsigned char *>(mmap(nullptr, size, prot, flags, fd_, 0));
    if (ptr_ == MAP_FAILED) {
      std::cerr << "Error: Failed memory mapping allocated file " << fname
                << " - " << strerror(errno) << std::endl;
      ptr_ = nullptr;
      close(fd_);
      fd_ = -1;
      return false;
    }

    // fd_ is kept open by allocate_and_map for writeable shared mappings
    return true;
  }

  /**
   * @brief Unmaps the memory region and closes the associated file descriptor.
   */
  void unmap() noexcept {
    if (ptr_ && size_ > 0) {
      munmap(ptr_, size_);
    }
    if (fd_ != -1) {
      close(fd_);
    }
    ptr_ = nullptr;
    size_ = 0;
    fd_ = -1;
  }

  unsigned char *get() const { return ptr_; }
  size_t size() const { return size_; }
  bool is_mapped() const { return ptr_ != nullptr; }
};

// --- Large File Header I/O ---

/**
 * @brief Reads and validates a large file header from an input file stream.
 *
 * This function reads a LargeFileHeader structure from the given input file
 * stream and performs validation checks on the magic number and version fields
 * to ensure the file format is supported.
 *
 * @param in_file Reference to the input file stream to read from
 * @param header Reference to LargeFileHeader structure to populate with read
 * data
 *
 * @return true if the header was successfully read and validated, false
 * otherwise
 *
 * @note Validation checks include:
 *       - Successful read operation
 *       - Magic number matches MAGIC_NUMBER_LARGE_FILE
 *       - Version is not greater than FORMAT_VERSION
 */
bool read_large_file_header(std::ifstream &in_file, LargeFileHeader &header) {
  in_file.read(reinterpret_cast<char *>(&header), sizeof(header));
  // Check magic number and version after reading
  if (!in_file || header.magic_number != MAGIC_NUMBER_LARGE_FILE ||
      header.version > FORMAT_VERSION) {
    // Don't print error here, allows caller to distinguish format types
    return false;
  }
  return true;
}

/**
 * @brief Reads block metadata from a binary file stream.
 *
 * This function reads a sequence of block sizes from the input file stream
 * and stores them in the provided vector. The block sizes are expected to
 * be stored as uint64_t values in binary format.
 *
 * @param in_file Reference to the input file stream to read from
 * @param block_sizes Vector to store the read block sizes (will be resized)
 * @param num_blocks Number of blocks to read metadata for
 * @return true if the read operation was successful, false otherwise
 *
 * @note The file stream position must be at the correct location before calling
 * this function
 */
bool read_block_metadata(std::ifstream &in_file,
                         std::vector<uint64_t> &block_sizes,
                         size_t num_blocks) {
  block_sizes.resize(num_blocks); // Resize to hold all block sizes
  in_file.read(reinterpret_cast<char *>(block_sizes.data()),
               num_blocks * sizeof(uint64_t));
  return in_file.good();
}

} // End anonymous namespace

//-----------------------------------------------------------------------------
// Public Interface Implementation
//-----------------------------------------------------------------------------

/**
 * @brief Processes a file for compression by determining the appropriate
 * compression strategy.
 *
 * This function analyzes the input file size and routes it to either the large
 * file or small file compression handler based on the configured threshold.
 *
 * @param input_path The path to the input file to be processed
 * @param cfg Configuration data containing compression settings and thresholds
 *
 * @return true if the file was successfully processed and compressed, false
 * otherwise
 */
bool process_file(const std::string &input_path, const ConfigData &cfg) {
  std::error_code ec;
  // Use std::filesystem::file_size for large file support
  uintmax_t input_size_uint = std::filesystem::file_size(input_path, ec);
  if (ec) {
    if (cfg.verbosity >= 1)
      std::cerr << "Error getting size for " << input_path << ": "
                << ec.message() << std::endl;
    return false;
  }
  size_t input_size =
      static_cast<size_t>(input_size_uint); // Cast for internal use

  if (input_size >= cfg.large_file_threshold) {
    return compress_large_file(input_path, input_size, cfg);
  } else {
    return compress_small_file(input_path, input_size, cfg);
  }
}

/**
 * @brief Decompresses a file by automatically detecting its format and calling
 * the appropriate decompression function.
 *
 * This function first attempts to determine if the input file is in large
 * format by reading and validating its header. If the file is identified as
 * large format, it delegates to decompress_large_file(). Otherwise, it
 * validates the file size for small format compatibility and delegates to
 * decompress_small_file().
 *
 * @param input_path The path to the compressed file to decompress
 * @param cfg Configuration data containing verbosity level and other settings
 *
 * @return true if decompression was successful, false otherwise
 *
 * @note The function performs format detection by:
 *       1. Attempting to read a large file header (checks magic number/version)
 *       2. If not large format, validates file size is at least
 * sizeof(uint64_t) for small format
 */
bool decompress_file(const std::string &input_path, const ConfigData &cfg) {
  std::ifstream test_file(input_path, std::ios::binary);
  if (!test_file) {
    if (cfg.verbosity >= 1)
      std::cerr << "Error opening compressed file: " << input_path << " ("
                << strerror(errno) << ")" << std::endl;
    return false;
  }

  LargeFileHeader header_test;
  // Try reading header - read_large_file_header checks magic/version
  bool is_large_format = read_large_file_header(test_file, header_test);
  test_file.close(); // Close the test handle

  if (is_large_format) {
    return decompress_large_file(input_path, cfg);
  } else {
    // If not large format, try small format (check if size > sizeof(uint64_t))
    std::error_code ec;
    uintmax_t file_size = std::filesystem::file_size(input_path, ec);
    if (ec || file_size < sizeof(uint64_t)) {
      if (cfg.verbosity >= 1)
        std::cerr << "Error: File " << input_path
                  << " is not a valid small or large compressed file format."
                  << std::endl;
      return false;
    }
    return decompress_small_file(input_path, cfg);
  }
}

//-----------------------------------------------------------------------------
// Internal Compression/Decompression Logic
//-----------------------------------------------------------------------------

/**
 * @brief Compresses a small file using miniz compression library.
 *
 * This function handles compression of files that can be entirely loaded into
 * memory. It memory-maps the input file, compresses the data using miniz's
 * compress() function, and writes the compressed output with a prepended 64-bit
 * original size header.
 *
 * @param input_path Path to the input file to be compressed
 * @param input_size Size of the input file in bytes (0 for auto-detection)
 * @param cfg Configuration data containing verbosity level and removal options
 *
 * @return true if compression succeeds and output file is written successfully,
 *         false if any error occurs during the process
 *
 * @details
 * - Memory-maps the input file for efficient reading
 * - Handles special case of 0-byte files by creating output with only size
 * header
 * - Uses miniz compressBound() to allocate sufficient buffer space
 * - Output format: [8-byte original size][compressed data]
 * - Output file has SUFFIX appended to original filename
 * - Optionally removes original file if cfg.remove_origin is true
 *
 * @note The function automatically unmaps the input file when MappedFile goes
 * out of scope
 * @warning If compression fails or file operations fail, partial output files
 * are cleaned up
 */
bool compress_small_file(const std::string &input_path, size_t input_size,
                         const ConfigData &cfg) {
  MappedFile mapped_in;
  size_t current_size = input_size; // map() might update if input_size was 0
  if (!mapped_in.map(input_path.c_str(), current_size, PROT_READ,
                     MAP_PRIVATE)) {
    if (cfg.verbosity >= 1)
      std::cerr << "Error mapping small input file: " << input_path << " ("
                << strerror(errno) << ")" << std::endl;
    return false;
  }
  input_size = current_size; // Use the actual size determined by map()

  // Handle 0-byte input file
  if (input_size == 0) {
    std::string output_path = input_path + SUFFIX;
    std::ofstream out_file(output_path, std::ios::binary | std::ios::trunc);
    if (!out_file) {
      if (cfg.verbosity >= 1)
        std::cerr << "Error creating empty output file for " << input_path
                  << std::endl;
      return false;
    }
    // Write only the original size (0)
    uint64_t zero_size = 0;
    out_file.write(reinterpret_cast<const char *>(&zero_size),
                   sizeof(zero_size));
    out_file.close();
    if (!out_file.good()) { // Check stream state after closing
      if (cfg.verbosity >= 1)
        std::cerr << "Error finalizing empty output file for " << input_path
                  << std::endl;
      std::filesystem::remove(output_path);
      return false;
    }
    if (cfg.remove_origin) {
      std::error_code ec;
      if (!std::filesystem::remove(input_path, ec) && cfg.verbosity >= 1) {
        std::cerr << "Warning: Failed to remove original file: " << input_path
                  << " - " << ec.message() << std::endl;
      }
    }
    return true;
  }

  // Here the file is completely mapped and has non-zero size

  unsigned char *in_ptr = mapped_in.get(); // gets the pointer to mapped data
  mz_ulong comp_len_bound =
      compressBound(input_size); // Get max compressed size
  std::vector<unsigned char> compressed_data(
      comp_len_bound);                       // Pre-allocate buffer
  mz_ulong actual_comp_len = comp_len_bound; // Actual size after compression

  // Simply call the miniz compress function
  int mz_result =
      compress(compressed_data.data(), &actual_comp_len, in_ptr, input_size);

  // mapped_in goes out of scope here and unmaps the input file

  if (mz_result != Z_OK) {
    if (cfg.verbosity >= 1)
      std::cerr << "Miniz compression failed for " << input_path
                << " with error: " << mz_error(mz_result) << std::endl;
    return false;
  }
  compressed_data.resize(actual_comp_len);

  std::string output_path = input_path + SUFFIX;
  std::ofstream out_file(output_path, std::ios::binary | std::ios::trunc);
  if (!out_file) {
    if (cfg.verbosity >= 1)
      std::cerr << "Error opening output file: " << output_path << std::endl;
    return false;
  }

  uint64_t orig_size_64 = static_cast<uint64_t>(input_size);
  out_file.write(reinterpret_cast<const char *>(&orig_size_64),
                 sizeof(orig_size_64));
  out_file.write(reinterpret_cast<const char *>(compressed_data.data()),
                 compressed_data.size());
  out_file.close();

  if (!out_file.good()) {
    if (cfg.verbosity >= 1)
      std::cerr << "Error writing to output file: " << output_path << std::endl;
    std::filesystem::remove(output_path);
    return false;
  }

  if (cfg.remove_origin) {
    std::error_code ec;
    if (!std::filesystem::remove(input_path, ec) && cfg.verbosity >= 1) {
      std::cerr << "Warning: Failed to remove original file: " << input_path
                << " - " << ec.message() << std::endl;
    }
  }
  return true;
}

/**
 * @brief Compresses a large file using parallel block-based compression with
 * memory mapping.
 *
 * This function performs efficient compression of large files by:
 * 1. Memory-mapping the input file for fast access
 * 2. Dividing the file into blocks and compressing them in parallel
 * 3. Writing the compressed output using memory mapping for optimal performance
 *
 * The compression process consists of three phases:
 * - Phase 1: Parallel compression of file blocks into memory buffers
 * - Phase 2: Sequential writing of compressed data to output file
 * - Phase 3: Cleanup of allocated resources
 *
 * @param input_path Path to the input file to be compressed
 * @param input_size Size of the input file in bytes
 * @param cfg Configuration data containing compression parameters:
 *            - block_size: Size of each compression block
 *            - num_threads: Number of parallel threads to use
 *            - verbosity: Level of diagnostic output
 *            - remove_origin: Whether to delete the original file after
 * compression
 *
 * @return true if compression succeeds, false if any error occurs
 *
 * @note The output file is created with the same name as input plus SUFFIX
 * extension
 * @note Uses zlib compression with default settings for each block
 * @note Memory-maps both input and output files for efficient I/O operations
 * @note Thread-safe compression using per-thread compressor states and buffers
 * @note Automatically removes incomplete output file on failure
 */
bool compress_large_file(const std::string &input_path, size_t input_size,
                         const ConfigData &cfg) {
  MappedFile mapped_in;             // RAII wrapper for input file mapping
  size_t current_size = input_size; // map() will confirm this size

  if (!mapped_in.map(input_path.c_str(), current_size, PROT_READ,
                     MAP_PRIVATE)) {
    if (cfg.verbosity >= 1)
      std::cerr << "Error mapping large input file: " << input_path << " ("
                << strerror(errno) << ")" << std::endl;
    return false;
  }
  if (current_size != input_size) {
    if (cfg.verbosity >= 1)
      std::cerr << "Error: File size changed during mapping for " << input_path
                << std::endl;
    return false;
  }

  unsigned char *in_ptr = mapped_in.get(); // Get pointer to mapped data

  // Calculate block count
  uint64_t num_blocks = (input_size + cfg.block_size - 1) / cfg.block_size;
  if (num_blocks == 0 && input_size > 0)
    num_blocks = 1; // Ensure at least one block if file not empty

  // --- Phase 1: Parallel Compression into Memory ---

  // Pre-allocate per-thread temporary buffers to avoid repeated vector
  // allocations
  int thread_count = cfg.num_threads;
  std::vector<std::vector<unsigned char>> thread_temp_buffers(
      thread_count); // Per-thread temp buffers
  for (int t = 0; t < thread_count; ++t) {
    // reserve max possible compressed size once per thread
    thread_temp_buffers[t].reserve(compressBound(cfg.block_size));
  }

  // Pre-allocate per-thread tdefl_compressor state for reuse and reduced init
  // overhead
  std::vector<tdefl_compressor *> thread_deflators(thread_count);
  for (int t = 0; t < thread_count; ++t) {
    thread_deflators[t] =
        tdefl_compressor_alloc(); // Allocate per-thread compressor
    // Initialize compressor state once per thread
    tdefl_init(thread_deflators[t], nullptr,
               nullptr, /* flags: default probes + zlib header */
               TDEFL_WRITE_ZLIB_HEADER | TDEFL_DEFAULT_MAX_PROBES);
  }

  std::vector<std::vector<unsigned char>> compressed_blocks_data(
      num_blocks); // Compressed data for each block
  std::vector<uint64_t> compressed_block_sizes(
      num_blocks); // Sizes of compressed blocks
  std::vector<int> block_mz_results(
      num_blocks, Z_OK); // Results of miniz compression for each block
  std::atomic<bool> compression_error_occurred =
      false; // Atomic flag to track any compression errors

// Phase 1: Parallel block read & compression using memory-mapped input
#pragma omp parallel for num_threads(cfg.num_threads) schedule(dynamic)
  for (uint64_t i = 0; i < num_blocks; ++i) {
    if (compression_error_occurred.load())
      continue;
    // Determine block range
    size_t offset = i * cfg.block_size;
    size_t blk_size =
        (i == num_blocks - 1) ? (input_size - offset) : cfg.block_size;
    const unsigned char *block_ptr = in_ptr + offset; // Pointer to block data

    int tid = omp_get_thread_num();
    tdefl_compressor *def =
        thread_deflators[tid]; // Get thread-specific compressor
    // Reset compressor state per block
    tdefl_init(def, nullptr, nullptr,
               TDEFL_WRITE_ZLIB_HEADER | TDEFL_DEFAULT_MAX_PROBES);
    auto &temp_buf =
        thread_temp_buffers[tid];          // Get thread-specific temp buffer
    size_t need = compressBound(blk_size); // Calculate max compressed size
    temp_buf.reserve(need);  // Ensure buffer can hold compressed data
    temp_buf.resize(need);   // Resize to max size
    mz_ulong out_len = need; // Output length for miniz compression
    int res = compress(temp_buf.data(), &out_len, block_ptr, blk_size);
    block_mz_results[i] = res; // Store result of miniz compression
    if (res == Z_OK) {         // Only proceed if compression succeeded
      compressed_block_sizes[i] = out_len;  // Store compressed size
      temp_buf.resize(out_len);             // Resize to actual compressed size
      compressed_blocks_data[i] = temp_buf; // Store compressed data
    } else {
      compression_error_occurred = true;
    }
  }

  // Input file can be unmapped now as all data processed
  mapped_in.unmap();

  // --- Phase 2: Check for Errors and Write Output File Sequentially ---
  bool success = !compression_error_occurred.load();
  if (success) {
    // Double check results (optional sanity check)
    for (uint64_t i = 0; i < num_blocks; ++i) {
      if (block_mz_results[i] !=
          Z_OK) { // If any block failed, set success to false
        if (cfg.verbosity >= 1)
          std::cerr << "Miniz compression failed for block " << i << " of "
                    << input_path << ": " << mz_error(block_mz_results[i])
                    << std::endl;
        success = false;
        break;
      }
    }
  } else {
    // Find and log the first error if not already logged inside loop
    for (uint64_t i = 0; i < num_blocks; ++i) { // Check all blocks
      if (block_mz_results[i] != Z_OK) { // If any block failed compression
        // Log the first error encountered
        if (cfg.verbosity >= 1)
          std::cerr << "Miniz compression failed for block " << i << " of "
                    << input_path << ": " << mz_error(block_mz_results[i])
                    << std::endl;
        break; // Log only the first error encountered
      }
    }
  }

  std::string output_path = input_path + SUFFIX; // Output file path with SUFFIX
  if (success) {
    // --- Phase 2: Writing Compressed Data with Metadata and File
    // It starts using allocated_and_map to create the output file at the final
    // dimension total_size It writes sequentially the header and metadata Then
    // it writes the compressed blocks in parallel
    {
      // Initialize header correctly
      LargeFileHeader header; // Default initialize (magic, version)
      header.original_size = static_cast<uint64_t>(input_size);
      header.num_blocks = num_blocks; // Number of compressed blocks

      size_t header_size = sizeof(header);
      size_t meta_size = num_blocks * sizeof(uint64_t);
      // Calculate total file size
      size_t total_size = header_size + meta_size;
      for (auto s : compressed_block_sizes)
        total_size += s;
      // Memory-map output file
      MappedFile mapped_out;
      if (!mapped_out.allocate_and_map(output_path.c_str(), total_size)) {
        if (cfg.verbosity >= 1)
          std::cerr << "Error allocating output mmap: " << output_path
                    << std::endl;
        success = false;
      } else {
        unsigned char *out_ptr = mapped_out.get(); // Mapped output pointer
        // Copy header and metadata
        memcpy(out_ptr, &header, header_size); // Copy header to output
        memcpy(out_ptr + header_size, compressed_block_sizes.data(),
               meta_size); // Copy metadata
        // Precompute output offsets for each block
        std::vector<size_t> data_offsets(num_blocks); // Offsets for each block
        size_t cur_offset =
            header_size + meta_size; // Start after header and metadata
        for (size_t i = 0; i < num_blocks; ++i) {  // Calculate offsets
          data_offsets[i] = cur_offset;            // Offset for this block
          cur_offset += compressed_block_sizes[i]; // Update for next block
        }
// Parallel copy compressed blocks into mapped region using precomputed offsets
#pragma omp parallel for num_threads(cfg.num_threads) schedule(dynamic)
        for (size_t i = 0; i < num_blocks; ++i) {
          size_t sz = compressed_block_sizes[i];
          if (sz > 0) {
            memcpy(out_ptr + data_offsets[i], compressed_blocks_data[i].data(),
                   sz); // Copy compressed block data to output
          }
        }
        // Unmap to flush
        mapped_out.unmap();
      }
    }
  }

  // --- Phase 3: Cleanup ---
  // Free allocated tdefl_compressor states and reset pointers
  for (int t = 0; t < thread_count; ++t) {
    if (thread_deflators[t]) {
      tdefl_compressor_free(thread_deflators[t]); // Free compressor state
      thread_deflators[t] = nullptr;              // Good practice
    }
  }

  if (!success) {
    std::filesystem::remove(output_path); // Remove partial/corrupt file
    return false;
  }

  // Optionally remove original
  if (cfg.remove_origin) {
    std::error_code ec;
    if (!std::filesystem::remove(input_path, ec) && cfg.verbosity >= 1) {
      std::cerr << "Warning: Failed to remove original file: " << input_path
                << " - " << ec.message() << std::endl;
    }
  }

  return true;
}

/**
 * @brief Decompresses a large file that was compressed using block-based
 * compression.
 *
 * This function implements a multi-phase decompression process for large files
 * that were compressed in blocks. It uses memory mapping and parallel
 * processing to efficiently decompress multiple blocks simultaneously.
 *
 * @param input_path Path to the compressed input file
 * @param cfg Configuration data containing decompression settings including:
 *            - verbosity: Level of logging output
 *            - num_threads: Number of threads for parallel decompression
 *            - block_size: Size of each decompression block
 *            - remove_origin: Whether to delete the input file after successful
 * decompression
 *
 * @return true if decompression was successful, false otherwise
 *
 * @details The function operates in four main phases:
 * 1. Header Reading: Sequentially reads the file header and block metadata
 * 2. File Mapping: Maps both input and output files to memory for efficient
 * access
 * 3. Parallel Decompression: Uses OpenMP to decompress blocks in parallel
 * 4. Validation and Cleanup: Verifies results and cleans up resources
 *
 * The function handles special cases including:
 * - Zero-byte files
 * - Empty blocks within the compressed data
 * - Memory mapping failures
 * - Decompression errors with detailed error reporting
 *
 * @note The function expects the input file to have a specific header format
 * with block metadata and uses miniz library for decompression.
 * @note Output file path is derived by removing the compression suffix or
 * adding ".out"
 */
bool decompress_large_file(const std::string &input_path,
                           const ConfigData &cfg) {
  // --- Phase 1: Read Header and Metadata Sequentially ---
  std::ifstream header_reader(input_path, std::ios::binary);
  if (!header_reader) {
    if (cfg.verbosity >= 1)
      std::cerr << "Error opening large compressed file: " << input_path << " ("
                << strerror(errno) << ")" << std::endl;
    return false;
  }

  LargeFileHeader header;
  if (!read_large_file_header(header_reader, header)) {
    if (cfg.verbosity >= 1)
      std::cerr << "Error: Invalid or unsupported large file header in: "
                << input_path << std::endl;
    return false;
  }
  if (header.num_blocks == 0 && header.original_size > 0) {
    if (cfg.verbosity >= 1)
      std::cerr << "Error: Header indicates zero blocks but non-zero original "
                   "size in: "
                << input_path << std::endl;
    return false;
  }

  std::vector<uint64_t> compressed_block_sizes; // Sizes of compressed blocks
  if (!read_block_metadata(header_reader, compressed_block_sizes,
                           header.num_blocks)) { // Read block sizes
    if (cfg.verbosity >= 1)
      std::cerr << "Error reading block metadata from: " << input_path
                << std::endl;
    return false;
  }

  // streampos is used to track position in the file
  std::streampos data_start_pos =
      header_reader.tellg(); // Position after metadata
  header_reader.close();     // Done with sequential reading

  // Determine output path
  std::string output_path = input_path; // Start with input path
  if (output_path.length() > SUFFIX.length() &&
      output_path.substr(output_path.length() - SUFFIX.length()) ==
          SUFFIX) { // Check for suffix
    output_path = output_path.substr(0, output_path.length() - SUFFIX.length());
  } else {
    if (cfg.verbosity >= 1)
      std::cerr << "Warning: Input file " << input_path
                << " does not have expected suffix " << SUFFIX
                << ". Appending .out" << std::endl;
    output_path += ".out";
  }

  // Handle 0-byte original file (no parallel processing needed)
  if (header.original_size == 0) {
    std::ofstream out_file(output_path, std::ios::binary | std::ios::trunc);
    if (!out_file) { /* ... error handling ... */
      return false;
    }
    out_file.close();
    if (!out_file.good()) { /* ... error handling ... */
      return false;
    }
    if (cfg.remove_origin) { /* ... remove original ... */
    }
    return true;
  }

  // --- Phase 2: Map Files and Calculate Offsets ---
  MappedFile mapped_in;            // RAII wrapper for input file mapping
  size_t compressed_file_size = 0; // Let map determine the size
  if (!mapped_in.map(input_path.c_str(), compressed_file_size, PROT_READ,
                     MAP_PRIVATE)) { // Memory-map input file
    if (cfg.verbosity >= 1)
      std::cerr << "Error mapping large compressed input file: " << input_path
                << std::endl;
    return false;
  }
  unsigned char *in_base_ptr = mapped_in.get(); // Base pointer to mapped input

  MappedFile mapped_out; // RAII wrapper for output file mapping
  if (!mapped_out.allocate_and_map(output_path.c_str(), header.original_size)) {
    if (cfg.verbosity >= 1)
      std::cerr << "Error allocating output file map for " << output_path
                << " size " << header.original_size << std::endl;
    // mapped_in unmapped by RAII
    std::filesystem::remove(output_path);
    return false;
  }
  // Ensure output map pointer is valid
  unsigned char *out_base_ptr = mapped_out.get();
  if (!out_base_ptr && header.original_size > 0) {
    if (cfg.verbosity >= 1)
      std::cerr << "Error: Output map pointer is null for non-zero size."
                << std::endl;
    std::filesystem::remove(output_path);
    return false;
  }

  // Calculate offsets for both compressed input and decompressed output
  std::vector<size_t> compressed_block_offsets(
      header.num_blocks); // Offsets for compressed blocks
  std::vector<size_t> decompressed_block_offsets(
      header.num_blocks); // Offsets for decompressed blocks
  size_t current_compressed_offset =
      static_cast<size_t>(data_start_pos); // Start after header and metadata
  size_t current_decompressed_offset = 0;

  for (uint64_t i = 0; i < header.num_blocks; ++i) {
    compressed_block_offsets[i] = current_compressed_offset;
    decompressed_block_offsets[i] = current_decompressed_offset;

    current_compressed_offset +=
        compressed_block_sizes[i]; // Update offset for next block

    size_t decomp_size_for_block =
        (i == header.num_blocks - 1)
            ? (header.original_size - current_decompressed_offset)
            : cfg.block_size; // Last block may be smaller
    current_decompressed_offset +=
        decomp_size_for_block; // Update output offset
  }
  // Sanity check total size after calculating offsets
  if (current_decompressed_offset != header.original_size) {
    if (cfg.verbosity >= 1)
      std::cerr << "Error: Calculated total decompressed size mismatch."
                << std::endl;
    return false; // mapped_in/out unmapped by RAII
  }
  if (current_compressed_offset > compressed_file_size) {
    if (cfg.verbosity >= 1)
      std::cerr << "Error: Calculated total compressed size exceeds file size."
                << std::endl;
    return false; // mapped_in/out unmapped by RAII
  }

  // --- Phase 3: Parallel Decompression ---
  std::atomic<bool> decompression_error = false; // Atomic flag for errors
  std::vector<int> block_mz_results_decomp(header.num_blocks,
                                           Z_OK); // Vector for Miniz results
  std::vector<mz_ulong> block_actual_decomp_sizes(
      header.num_blocks, 0); // Actual decompressed sizes for each block

#pragma omp parallel for num_threads(cfg.num_threads) schedule(dynamic)
  for (uint64_t i = 0; i < header.num_blocks; ++i) {
    if (decompression_error.load()) {
      continue; // Skip if an error occurred elsewhere
    }

    size_t compressed_size = compressed_block_sizes[i]; // Size of this block
    size_t expected_decomp_size =
        (i == header.num_blocks - 1)
            ? (header.original_size - decompressed_block_offsets[i])
            : cfg.block_size; // Expected size for this block

    // Handle empty block case
    if (compressed_size == 0) {
      if (expected_decomp_size != 0) {
        // Error: compressed is 0 but expected is not
        block_mz_results_decomp[i] = Z_DATA_ERROR; // Indicate error
        decompression_error = true;
      } else {
        block_mz_results_decomp[i] = Z_OK; // Correctly handled empty block
        block_actual_decomp_sizes[i] = 0;
      }
      continue; // Move to next block
    }

    // Get pointers for this block
    const unsigned char *compressed_block_ptr =
        in_base_ptr + compressed_block_offsets[i]; // Compressed data pointer
    unsigned char *decompressed_block_ptr =
        out_base_ptr + decompressed_block_offsets[i]; // Output pointer

    // Decompress directly into the output map
    mz_ulong dest_len = expected_decomp_size; // Pass expected size
    int mz_result = Z_OK;                     // Default to success

    // Check pointers before calling uncompress (defensive)
    if (!decompressed_block_ptr && expected_decomp_size > 0) {
      mz_result = Z_MEM_ERROR; // Indicate output pointer issue
    } else if (!compressed_block_ptr) {
      mz_result = Z_MEM_ERROR; // Indicate input pointer issue
    } else {
      mz_result = uncompress(decompressed_block_ptr, &dest_len,
                             compressed_block_ptr, compressed_size);
    }

    // Store results and check for errors
    block_mz_results_decomp[i] = mz_result;
    block_actual_decomp_sizes[i] = dest_len;

    if (mz_result != Z_OK || dest_len != expected_decomp_size) {
      decompression_error = true;
    }
  } // --- End of parallel for ---

  // --- Phase 4: Final Checks and Cleanup ---
  mapped_in.unmap();  // Input file no longer needed
  mapped_out.unmap(); // Ensure output is flushed to disk

  bool success = !decompression_error.load();
  size_t final_decompressed_size = 0;

  if (success) {
    // Verify results sequentially after parallel execution
    for (uint64_t i = 0; i < header.num_blocks; ++i) {
      size_t expected_decomp_size =
          (i == header.num_blocks - 1)
              ? (header.original_size - decompressed_block_offsets[i])
              : cfg.block_size;

      if (block_mz_results_decomp[i] != Z_OK) {
        if (cfg.verbosity >= 1)
          std::cerr << "Miniz decompression failed for block " << i << " of "
                    << input_path << ": "
                    << mz_error(block_mz_results_decomp[i]) << std::endl;
        success = false;
        break;
      }
      if (block_actual_decomp_sizes[i] != expected_decomp_size) {
        if (cfg.verbosity >= 1)
          std::cerr << "Decompression size mismatch for block " << i << " of "
                    << input_path << ": expected " << expected_decomp_size
                    << ", got " << block_actual_decomp_sizes[i] << std::endl;
        success = false;
        break;
      }
      final_decompressed_size += block_actual_decomp_sizes[i];
    }
    // Final size check
    if (success && final_decompressed_size != header.original_size) {
      if (cfg.verbosity >= 1)
        std::cerr << "Error: Final total decompressed size ("
                  << final_decompressed_size
                  << ") does not match header original size ("
                  << header.original_size << ") for " << input_path
                  << std::endl;
      success = false;
    }

  } else {
    // Find and log the first error if not logged inside loop
    for (uint64_t i = 0; i < header.num_blocks; ++i) {
      size_t expected_decomp_size =
          (i == header.num_blocks - 1)
              ? (header.original_size - decompressed_block_offsets[i])
              : cfg.block_size;
      if (block_mz_results_decomp[i] != Z_OK) {
        if (cfg.verbosity >= 1)
          std::cerr << "Miniz decompression failed for block " << i << " of "
                    << input_path << ": "
                    << mz_error(block_mz_results_decomp[i]) << std::endl;
        break;
      }
      if (block_actual_decomp_sizes[i] != expected_decomp_size &&
          compressed_block_sizes[i] !=
              0) { // Don't log size mismatch for correctly handled empty blocks
        if (cfg.verbosity >= 1)
          std::cerr << "Decompression size mismatch for block " << i << " of "
                    << input_path << ": expected " << expected_decomp_size
                    << ", got " << block_actual_decomp_sizes[i] << std::endl;
        break;
      }
    }
  }

  if (!success) {
    std::filesystem::remove(output_path); // Remove potentially corrupt output
    return false;
  }

  // Optionally remove original compressed file
  if (cfg.remove_origin) {
    std::error_code ec;
    if (!std::filesystem::remove(input_path, ec) && cfg.verbosity >= 1) {
      std::cerr << "Warning: Failed to remove compressed file: " << input_path
                << " - " << ec.message() << std::endl;
    }
  }

  return true;
}

/**
 * @brief Decompresses a small compressed file using memory mapping and miniz
 * library.
 *
 * This function handles the decompression of files that can fit entirely in
 * memory. It uses memory mapping for efficient I/O operations and the miniz
 * library for decompression. The function expects the compressed file to
 * contain an 8-byte header with the original file size followed by the
 * compressed data.
 *
 * @param input_path Path to the compressed input file to decompress
 * @param cfg Configuration data containing verbosity level and removal options
 *
 * @return true if decompression succeeds, false otherwise
 *
 * @details
 * The function performs the following operations:
 * - Maps the compressed file into memory for reading
 * - Reads the original file size from the 8-byte header
 * - Determines the output file path by removing the compression suffix or
 * adding .out
 * - Handles special case of zero-byte original files
 * - Allocates and maps the output file with the original size
 * - Decompresses the data using miniz uncompress function
 * - Validates the decompressed size matches the expected original size
 * - Optionally removes the original compressed file if cfg.remove_origin is
 * true
 */
bool decompress_small_file(const std::string &input_path,
                           const ConfigData &cfg) {
  MappedFile mapped_in;
  size_t compressed_size = 0;
  std::error_code ec;
  uintmax_t file_size_uint = std::filesystem::file_size(input_path, ec);
  if (ec) {
    if (cfg.verbosity >= 1)
      std::cerr << "Error: Cannot stat compressed file " << input_path << " - "
                << ec.message() << std::endl;
    return false;
  }
  compressed_size = static_cast<size_t>(file_size_uint);

  if (compressed_size <
      sizeof(uint64_t)) { // Must contain at least the original size
    if (cfg.verbosity >= 1)
      std::cerr << "Error: Compressed file is too small: " << input_path
                << std::endl;
    return false;
  }

  if (!mapped_in.map(input_path.c_str(), compressed_size, PROT_READ,
                     MAP_PRIVATE)) {
    if (cfg.verbosity >= 1)
      std::cerr << "Error mapping small compressed file: " << input_path << " ("
                << strerror(errno) << ")" << std::endl;
    return false;
  }

  if (mapped_in.size() != compressed_size) {
    if (cfg.verbosity >= 1)
      std::cerr << "Error: Mapped size (" << mapped_in.size()
                << ") mismatch stat size (" << compressed_size << ") for "
                << input_path << std::endl;
    return false; // Should not happen if map() uses fstat correctly
  }

  unsigned char *in_ptr = mapped_in.get();
  uint64_t original_size = 0;
  // Check if we can actually read the size header
  if (compressed_size < sizeof(original_size)) {
    if (cfg.verbosity >= 1)
      std::cerr << "Error: File " << input_path
                << " too small to contain size header." << std::endl;
    return false; // mapped_in RAII handles unmap
  }
  memcpy(&original_size, in_ptr, sizeof(original_size));

  unsigned char *compressed_data_ptr = in_ptr + sizeof(original_size);
  size_t compressed_data_size = compressed_size - sizeof(original_size);

  std::string output_path = input_path;
  if (output_path.length() > SUFFIX.length() &&
      output_path.substr(output_path.length() - SUFFIX.length()) == SUFFIX) {
    output_path = output_path.substr(0, output_path.length() - SUFFIX.length());
  } else {
    if (cfg.verbosity >= 1)
      std::cerr << "Warning: Input file " << input_path
                << " does not have expected suffix " << SUFFIX
                << ". Appending .out" << std::endl;
    output_path += ".out";
  }

  // Handle 0-byte original file
  if (original_size == 0) {
    std::ofstream out_file(output_path, std::ios::binary | std::ios::trunc);
    if (!out_file) {
      if (cfg.verbosity >= 1)
        std::cerr << "Error creating empty decompressed file for " << input_path
                  << std::endl;
      return false;
    }
    out_file.close();
    if (!out_file.good()) {
      if (cfg.verbosity >= 1)
        std::cerr << "Error finalizing empty decompressed file for "
                  << input_path << std::endl;
      std::filesystem::remove(output_path);
      return false;
    }
    // mapped_in scope ends, unmaps input
    if (cfg.remove_origin) {
      std::error_code ec_rem;
      if (!std::filesystem::remove(input_path, ec_rem) && cfg.verbosity >= 1) {
        std::cerr << "Warning: Failed to remove original file: " << input_path
                  << " - " << ec_rem.message() << std::endl;
      }
    }
    return true;
  }

  // Allocate output file and map it
  MappedFile mapped_out;
  if (!mapped_out.allocate_and_map(output_path.c_str(), original_size)) {
    if (cfg.verbosity >= 1)
      std::cerr << "Error allocating output file map for " << output_path
                << std::endl;
    std::filesystem::remove(output_path); // Cleanup allocation attempt
    return false;
  }
  unsigned char *out_ptr = mapped_out.get();
  if (!out_ptr &&
      original_size > 0) { // Check if mapping failed for non-empty file
    if (cfg.verbosity >= 1)
      std::cerr << "Error: Output file mapping resulted in null pointer for "
                   "non-zero size "
                << original_size << std::endl;
    std::filesystem::remove(output_path);
    return false;
  }
  mz_ulong dest_len = original_size; // uncompress expects ulong

  // Decompress
  int mz_result = Z_OK;
  if (original_size >
      0) { // Only call uncompress if there's data to decompress/write
    mz_result = uncompress(out_ptr, &dest_len, compressed_data_ptr,
                           compressed_data_size);
  } else {
    dest_len = 0; // If original size was 0, dest_len should be 0
  }

  // mapped_in scope ends here, unmaps input
  // mapped_out scope ends *after* check, unmaps output

  if (mz_result != Z_OK) {
    if (cfg.verbosity >= 1)
      std::cerr << "Miniz decompression failed for " << input_path << ": "
                << mz_error(mz_result) << std::endl;
    std::filesystem::remove(output_path);
    return false;
  }
  if (dest_len != original_size) {
    if (cfg.verbosity >= 1)
      std::cerr << "Decompression size mismatch for " << input_path
                << ": expected " << original_size << ", got " << dest_len
                << std::endl;
    std::filesystem::remove(output_path);
    return false;
  }

  // Explicitly unmap output before potentially removing input
  mapped_out.unmap();

  if (cfg.remove_origin) {
    std::error_code ec_rem;
    if (!std::filesystem::remove(input_path, ec_rem) && cfg.verbosity >= 1) {
      std::cerr << "Warning: Failed to remove compressed file: " << input_path
                << " - " << ec_rem.message() << std::endl;
    }
  }
  return true;
}

} // namespace Compressor
