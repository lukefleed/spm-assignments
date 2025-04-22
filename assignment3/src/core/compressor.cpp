#include "compressor.hpp"
#include "miniz.h" // Include the actual miniz implementation

#include <cstring> // For strerror
#include <fcntl.h> // For open
#include <filesystem>
#include <fstream>
#include <iostream>     // For errors/verbose output
#include <memory>       // For unique_ptr
#include <omp.h>        // For parallel block compression
#include <sys/mman.h>   // For mmap/munmap
#include <sys/stat.h>   // For file stats, fstat
#include <system_error> // For error codes
#include <unistd.h>     // For close, ftruncate, unlink
#include <vector>

namespace Compressor {

//-----------------------------------------------------------------------------
// Internal Helper Functions (Anonymous Namespace)
//-----------------------------------------------------------------------------
namespace { // Start anonymous namespace

// --- Memory Mapping Utilities ---

/** @brief RAII wrapper for memory-mapped file pointers. */
class MappedFile {
  unsigned char *ptr_ = nullptr;
  size_t size_ = 0;
  int fd_ = -1; // File descriptor if opened by mapFile

public:
  MappedFile() = default;
  // Non-copyable
  MappedFile(const MappedFile &) = delete;
  MappedFile &operator=(const MappedFile &) = delete;
  // Movable
  MappedFile(MappedFile &&other) noexcept
      : ptr_(other.ptr_), size_(other.size_), fd_(other.fd_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
    other.fd_ = -1;
  }
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

  ~MappedFile() { unmap(); }

  bool map(const char *fname, size_t &size_in_out, int prot, int flags,
           int open_flags = O_RDONLY) {
    unmap(); // Ensure previous mapping is released

    fd_ = open(fname, open_flags);
    if (fd_ < 0) {
      std::cerr << "Error: Failed opening file " << fname << " - "
                << strerror(errno) << std::endl;
      return false;
    }

    if (size_in_out == 0) {
      struct stat s;
      if (fstat(fd_, &s) != 0) {
        std::cerr << "Error: Failed to fstat file " << fname << " - "
                  << strerror(errno) << std::endl;
        close(fd_);
        fd_ = -1;
        return false;
      }
      size_in_out = s.st_size;
      if (size_in_out == 0) { // Handle empty file case
        close(fd_);
        fd_ = -1;
        size_ = 0;
        ptr_ = nullptr; // No mapping needed for empty file
        return true;
      }
    }

    ptr_ = static_cast<unsigned char *>(
        mmap(nullptr, size_in_out, prot, flags, fd_, 0));
    if (ptr_ == MAP_FAILED) {
      std::cerr << "Error: Failed to memory map file " << fname << " - "
                << strerror(errno) << std::endl;
      ptr_ = nullptr;
      close(fd_);
      fd_ = -1;
      return false;
    }
    size_ = size_in_out;
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

  bool allocate_and_map(const char *fname, size_t size,
                        int prot = PROT_READ | PROT_WRITE,
                        int flags = MAP_SHARED) {
    unmap();
    size_ = size; // Store intended size

    fd_ = open(fname, O_RDWR | O_CREAT | O_TRUNC, 0666);
    if (fd_ < 0) {
      std::cerr << "Error: Failed creating/opening output file " << fname
                << " - " << strerror(errno) << std::endl;
      return false;
    }

    if (size > 0) { // ftruncate fails for size 0 sometimes
      if (ftruncate(fd_, size) < 0) {
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

  void unmap() {
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

bool write_large_file_header(std::ofstream &out_file,
                             const LargeFileHeader &header) {
  out_file.write(reinterpret_cast<const char *>(&header), sizeof(header));
  return out_file.good();
}

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

bool write_block_metadata(std::ofstream &out_file,
                          const std::vector<uint64_t> &block_sizes) {
  out_file.write(reinterpret_cast<const char *>(block_sizes.data()),
                 block_sizes.size() * sizeof(uint64_t));
  return out_file.good();
}

bool read_block_metadata(std::ifstream &in_file,
                         std::vector<uint64_t> &block_sizes,
                         size_t num_blocks) {
  block_sizes.resize(num_blocks);
  in_file.read(reinterpret_cast<char *>(block_sizes.data()),
               num_blocks * sizeof(uint64_t));
  return in_file.good();
}

} // End anonymous namespace

//-----------------------------------------------------------------------------
// Public Interface Implementation
//-----------------------------------------------------------------------------

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

// compress_small_file(...) implementation (come nella risposta precedente)
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

  unsigned char *in_ptr = mapped_in.get();
  mz_ulong comp_len_bound = compressBound(input_size);
  std::vector<unsigned char> compressed_data(comp_len_bound);
  mz_ulong actual_comp_len = comp_len_bound;

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

// --- Implementazione SEQUENZIALE per compress_large_file ---
bool compress_large_file(const std::string &input_path, size_t input_size,
                         const ConfigData &cfg) {
  MappedFile mapped_in;
  size_t current_size = input_size; // map() will confirm this size
  if (!mapped_in.map(input_path.c_str(), current_size, PROT_READ,
                     MAP_PRIVATE)) {
    if (cfg.verbosity >= 1)
      std::cerr << "Error mapping large input file: " << input_path << " ("
                << strerror(errno) << ")" << std::endl;
    return false;
  }
  // Ensure the size didn't somehow change (unlikely but defensive)
  if (current_size != input_size) {
    if (cfg.verbosity >= 1)
      std::cerr << "Error: File size changed during mapping for " << input_path
                << std::endl;
    return false; // mapped_in RAII handles unmap
  }

  unsigned char *in_ptr = mapped_in.get();

  // Calculate block count
  uint64_t num_blocks = (input_size + cfg.block_size - 1) / cfg.block_size;
  if (num_blocks == 0 && input_size > 0)
    num_blocks = 1; // Ensure at least one block if file not empty

  std::string output_path = input_path + SUFFIX;
  std::ofstream out_file(output_path, std::ios::binary | std::ios::trunc);
  if (!out_file) {
    if (cfg.verbosity >= 1)
      std::cerr << "Error opening output file: " << output_path << " ("
                << strerror(errno) << ")" << std::endl;
    return false;
  }

  // --- Phase 1: Write Header Placeholder ---
  LargeFileHeader header;
  header.original_size = input_size;
  header.num_blocks = num_blocks;
  if (!write_large_file_header(out_file, header)) {
    if (cfg.verbosity >= 1)
      std::cerr << "Error writing header to: " << output_path << std::endl;
    out_file.close();
    std::filesystem::remove(output_path);
    return false;
  }

  // --- Phase 2: Write Metadata Placeholder ---
  std::vector<uint64_t> compressed_block_sizes(num_blocks, 0); // Initialize
  std::streampos metadata_start_pos = out_file.tellp();
  if (!write_block_metadata(out_file, compressed_block_sizes)) {
    if (cfg.verbosity >= 1)
      std::cerr << "Error writing placeholder metadata to: " << output_path
                << std::endl;
    out_file.close();
    std::filesystem::remove(output_path);
    return false;
  }
  std::streampos data_start_pos = out_file.tellp();

  // --- Phase 3: Compress Blocks and Write Data (SEQUENTIAL) ---
  // Prepare buffer for compressed data (reuse is efficient)
  mz_ulong comp_len_bound_per_block = compressBound(cfg.block_size);
  std::vector<unsigned char> temp_compressed_block(comp_len_bound_per_block);
  bool success = true;

  for (uint64_t i = 0; i < num_blocks; ++i) {
    size_t block_offset = i * cfg.block_size;
    size_t block_size =
        (i == num_blocks - 1) ? (input_size - block_offset) : cfg.block_size;
    const unsigned char *block_start_ptr = in_ptr + block_offset;

    // Ensure buffer is large enough (should be unless block_size > initial
    // bound calc)
    if (block_size > 0 &&
        compressBound(block_size) > temp_compressed_block.size()) {
      // This should ideally not happen if block_size is constant, but handle
      // defensively
      temp_compressed_block.resize(compressBound(block_size));
    }

    mz_ulong actual_comp_len =
        temp_compressed_block.size(); // Pass buffer size as limit
    int mz_result = Z_OK;

    if (block_size > 0) { // Only compress if block has data
      mz_result = compress(temp_compressed_block.data(), &actual_comp_len,
                           block_start_ptr, block_size);
    } else {
      actual_comp_len = 0; // Zero-byte block compresses to zero bytes
    }

    if (mz_result != Z_OK) {
      if (cfg.verbosity >= 1)
        std::cerr << "Miniz compression failed for block " << i << " of "
                  << input_path << ": " << mz_error(mz_result) << std::endl;
      success = false;
      break; // Exit loop on first error
    }

    // Store compressed size
    compressed_block_sizes[i] = actual_comp_len;

    // Write compressed data (if any)
    if (actual_comp_len > 0) {
      out_file.write(
          reinterpret_cast<const char *>(temp_compressed_block.data()),
          actual_comp_len);
      if (!out_file) {
        if (cfg.verbosity >= 1)
          std::cerr << "Error writing compressed block " << i << " to "
                    << output_path << std::endl;
        success = false;
        break;
      }
    }
  } // End sequential block loop

  // --- Phase 4: Update Metadata if successful ---
  if (success) {
    out_file.seekp(metadata_start_pos); // Go back to metadata position
    if (!write_block_metadata(out_file, compressed_block_sizes)) {
      if (cfg.verbosity >= 1)
        std::cerr << "Error updating block metadata in: " << output_path
                  << std::endl;
      success = false;
    }
  }

  // --- Phase 5: Cleanup ---
  out_file.close();  // Close the file regardless of success/failure
  mapped_in.unmap(); // Input file unmapped by RAII

  if (!success ||
      !out_file.good()) { // Check success flag and final stream state
    if (cfg.verbosity >= 1 && success)
      std::cerr << "Error during final write/close for: " << output_path
                << std::endl;             // Log error if it happened on close
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

// --- Implementazione SEQUENZIALE per decompress_large_file ---
bool decompress_large_file(const std::string &input_path,
                           const ConfigData &cfg) {
  std::ifstream in_file(input_path, std::ios::binary);
  if (!in_file) {
    if (cfg.verbosity >= 1)
      std::cerr << "Error opening large compressed file: " << input_path << " ("
                << strerror(errno) << ")" << std::endl;
    return false;
  }

  // Read and validate header
  LargeFileHeader header;
  if (!read_large_file_header(in_file, header)) {
    // Don't print error here, let decompress_file handle 'not large format'
    // case if needed But if it WAS supposed to be large format, this is an
    // error. Let's assume if we reach here, it's expected to be large format.
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

  // Read block metadata (compressed sizes)
  std::vector<uint64_t> compressed_block_sizes;
  if (!read_block_metadata(in_file, compressed_block_sizes,
                           header.num_blocks)) {
    if (cfg.verbosity >= 1)
      std::cerr << "Error reading block metadata from: " << input_path
                << std::endl;
    return false;
  }
  std::streampos data_start_pos = in_file.tellg(); // Position after metadata

  // Determine output path
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
  if (header.original_size == 0) {
    std::ofstream out_file(output_path, std::ios::binary | std::ios::trunc);
    if (!out_file) {
      if (cfg.verbosity >= 1)
        std::cerr << "Error creating empty decompressed file for " << input_path
                  << std::endl;
      return false;
    }
    out_file.close();
    in_file.close(); // Close input handle
    if (!out_file.good()) {
      if (cfg.verbosity >= 1)
        std::cerr << "Error finalizing empty decompressed file for "
                  << input_path << std::endl;
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

  // Allocate output file and map it
  MappedFile mapped_out;
  if (!mapped_out.allocate_and_map(output_path.c_str(), header.original_size)) {
    if (cfg.verbosity >= 1)
      std::cerr << "Error allocating output file map for " << output_path
                << " size " << header.original_size << std::endl;
    in_file.close();
    std::filesystem::remove(output_path); // Cleanup allocation attempt
    return false;
  }
  unsigned char *out_ptr = mapped_out.get();

  // Prepare buffer for reading compressed blocks
  size_t max_comp_block_size = 0;
  for (uint64_t comp_size : compressed_block_sizes) {
    if (comp_size > max_comp_block_size)
      max_comp_block_size = comp_size;
  }
  // Allocate only if max > 0 to handle case where all blocks might be 0-byte
  std::vector<unsigned char> compressed_block_buffer;
  if (max_comp_block_size > 0) {
    compressed_block_buffer.resize(max_comp_block_size);
  }

  // Decompress blocks sequentially
  size_t current_output_offset = 0;
  bool success = true;
  in_file.seekg(data_start_pos); // Ensure we start reading data after metadata

  for (uint64_t i = 0; i < header.num_blocks; ++i) {
    size_t expected_decomp_size =
        (i == header.num_blocks - 1)
            ? (header.original_size - current_output_offset)
            : cfg.block_size;
    size_t compressed_size = compressed_block_sizes[i];

    // Handle empty compressed block
    if (compressed_size == 0) {
      if (expected_decomp_size != 0) {
        if (cfg.verbosity >= 1)
          std::cerr
              << "Error: Block " << i
              << " compressed size is 0, but expected decompressed size is "
              << expected_decomp_size << " in " << input_path << std::endl;
        success = false;
        break;
      }
      // If expected is also 0, just continue to next block
      continue;
    }

    // Ensure buffer is large enough (should be due to pre-allocation)
    if (compressed_size > compressed_block_buffer.size()) {
      if (cfg.verbosity >= 1)
        std::cerr << "Error: Internal buffer too small for compressed block "
                  << i << " of size " << compressed_size << std::endl;
      success = false;
      break;
    }

    // Read compressed block
    in_file.read(reinterpret_cast<char *>(compressed_block_buffer.data()),
                 compressed_size);
    if (!in_file) {
      if (cfg.verbosity >= 1)
        std::cerr << "Error reading compressed block " << i << " (size "
                  << compressed_size << ") from " << input_path << std::endl;
      success = false;
      break;
    }

    // Decompress into the correct offset in the output map
    mz_ulong dest_len = expected_decomp_size; // Pass expected size
    int mz_result = Z_OK;

    // Check if output pointer is valid before calling uncompress
    if (out_ptr == nullptr && expected_decomp_size > 0) {
      if (cfg.verbosity >= 1)
        std::cerr << "Error: Output buffer is null but expected decompressed "
                     "size is non-zero for block "
                  << i << std::endl;
      success = false;
      break;
    }

    if (expected_decomp_size >
        0) { // Only call uncompress if there's data expected
      mz_result = uncompress(out_ptr + current_output_offset, &dest_len,
                             compressed_block_buffer.data(), compressed_size);
    } else {
      dest_len = 0; // If expected size is 0, actual decompressed size is 0
    }

    if (mz_result != Z_OK) {
      if (cfg.verbosity >= 1)
        std::cerr << "Miniz decompression failed for block " << i << " of "
                  << input_path << ": " << mz_error(mz_result) << std::endl;
      success = false;
      break;
    }
    if (dest_len != expected_decomp_size) {
      if (cfg.verbosity >= 1)
        std::cerr << "Decompression size mismatch for block " << i << " of "
                  << input_path << ": expected " << expected_decomp_size
                  << ", got " << dest_len << std::endl;
      success = false;
      break;
    }

    current_output_offset += dest_len;
  } // End sequential block loop

  // Cleanup input stream first
  in_file.close();

  // Unmap output file (RAII handles this, but can be explicit if needed before
  // final checks)
  mapped_out.unmap();

  // Final checks
  if (success && current_output_offset != header.original_size) {
    if (cfg.verbosity >= 1)
      std::cerr << "Error: Final decompressed size (" << current_output_offset
                << ") does not match header original size ("
                << header.original_size << ") for " << input_path << std::endl;
    success = false;
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

// decompress_small_file(...) implementation (come nella risposta precedente)
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
