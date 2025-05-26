#!/usr/bin/env python3

import argparse
import struct
import random
import sys
import os

# Define the maximum payload size for the Record struct in Python, mirroring C++
# This isn't strictly needed for generation logic if we only write args.payload_bytes,
# but good for consistency if we were to read/manipulate full MAX_RPAYLOAD_SIZE buffers.
MAX_RPAYLOAD_SIZE_PY = 256

def generate_records(num_elements, payload_bytes, seed, distribution):
    """
    Generates records based on the specified parameters.

    Args:
        num_elements (int): Number of records to generate.
        payload_bytes (int): Size of the payload for each record in bytes.
        seed (int): Seed for the random number generator.
        distribution (str): The distribution type for keys.

    Yields:
        tuple: (key, payload_bytes_array) for each record.
    """
    if payload_bytes < 0 or payload_bytes > MAX_RPAYLOAD_SIZE_PY:
        raise ValueError(f"Payload size must be between 0 and {MAX_RPAYLOAD_SIZE_PY}. Got {payload_bytes}")

    # Max value for an unsigned long key (64-bit)
    max_key_val = (1 << 64) - 1

    random.seed(seed)

    if distribution == "random":
        for _ in range(num_elements):
            key = random.randint(0, max_key_val)
            payload = bytearray(random.getrandbits(8) for _ in range(payload_bytes))
            yield key, payload
    elif distribution == "sorted_asc":
        for i in range(num_elements):
            key = i
            # Consistent payload generation for deterministic testing if needed
            # For simplicity, random payload is fine here too unless specific payload patterns are tested.
            payload_content_base = (i % 256)
            payload = bytearray((payload_content_base + j) % 256 for j in range(payload_bytes))
            yield key, payload
    elif distribution == "sorted_desc":
        for i in range(num_elements):
            key = num_elements - 1 - i
            payload_content_base = ((num_elements - 1 - i) % 256)
            payload = bytearray((payload_content_base + j) % 256 for j in range(payload_bytes))
            yield key, payload
    elif distribution == "few_unique":
        num_unique_keys = max(1, num_elements // 1000) # Ensure at least one unique key
        if num_elements > 0 and num_unique_keys == 0: # Handle very small N
             num_unique_keys = 1

        unique_keys = sorted([random.randint(0, max_key_val) for _ in range(num_unique_keys)])
        if not unique_keys and num_elements > 0 : # if num_unique_keys was 0 and N > 0
            unique_keys = [random.randint(0, max_key_val)]


        for i in range(num_elements):
            key = unique_keys[i % len(unique_keys)] if unique_keys else 0
            payload_content_base = (key % 256)
            payload = bytearray((payload_content_base + j) % 256 for j in range(payload_bytes))
            yield key, payload
    elif distribution == "all_equal":
        key_val = random.randint(0, max_key_val // 2) # Keep it somewhat in range
        for _ in range(num_elements):
            payload = bytearray(random.getrandbits(8) for _ in range(payload_bytes))
            yield key_val, payload
    elif distribution == "mostly_sorted":
        # Generate sorted_asc first
        records_list = []
        for i in range(num_elements):
            key = i
            payload_content_base = (i % 256)
            payload = bytearray((payload_content_base + j) % 256 for j in range(payload_bytes))
            records_list.append({'key': key, 'payload': payload})

        # Swap a small percentage of elements
        num_swaps = max(1, num_elements // 20) # 5% of elements, ensure at least one swap for small N
        if num_elements < 2: # Cannot swap if less than 2 elements
            num_swaps = 0

        for _ in range(num_swaps):
            idx1 = random.randint(0, num_elements - 1)
            idx2 = random.randint(0, num_elements - 1)
            if idx1 == idx2 and num_elements > 1: # Ensure different indices if possible
                idx2 = (idx1 + 1) % num_elements
            if num_elements > 1 : # only swap if possible
                records_list[idx1], records_list[idx2] = records_list[idx2], records_list[idx1]

        for record in records_list:
            yield record['key'], record['payload']
    else:
        raise ValueError(f"Unknown distribution type: {distribution}")

def parse_size_string(size_str):
    """ Parses size string like '10M', '100K' into an integer. """
    size_str_lower = size_str.lower()
    multiplier = 1
    if size_str_lower.endswith('m'):
        multiplier = 1000 * 1000
        size_str_lower = size_str_lower[:-1]
    elif size_str_lower.endswith('k'):
        multiplier = 1000
        size_str_lower = size_str_lower[:-1]

    try:
        return int(size_str_lower) * multiplier
    except ValueError:
        raise ValueError(f"Invalid size string: {size_str}")


def main():
    parser = argparse.ArgumentParser(description="Generate binary data file of Records.")
    parser.add_argument("--size", type=str, required=True, help="Number of records (e.g., 10M, 100K, or 1000000).")
    parser.add_argument("--payload", type=int, required=True, help="Record payload size in bytes.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output binary file.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random data generation.")
    parser.add_argument("--distribution", type=str, default="random",
                        choices=["random", "sorted_asc", "sorted_desc", "few_unique", "all_equal", "mostly_sorted"],
                        help="Key distribution type.")

    args = parser.parse_args()

    try:
        num_elements = parse_size_string(args.size)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.payload < 0 :
        print(f"Error: Payload size cannot be negative. Got {args.payload}", file=sys.stderr)
        sys.exit(1)
    if args.payload > MAX_RPAYLOAD_SIZE_PY:
        print(f"Warning: Requested payload size {args.payload} exceeds MAX_RPAYLOAD_SIZE_PY {MAX_RPAYLOAD_SIZE_PY}. "
              f"Will use {args.payload}, but C++ struct might truncate or behave unexpectedly if it's smaller.", file=sys.stderr)
        # The script will generate args.payload bytes. C++ side will handle based on its MAX_RPAYLOAD_SIZE
        # and the runtime r_payload_size.

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            print(f"Error: Could not create output directory {output_dir}: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        with open(args.output, "wb") as f_out:
            for key, payload_data in generate_records(num_elements, args.payload, args.seed, args.distribution):
                # struct Record { unsigned long key; char rpayload[RPAYLOAD]; };
                # Pack key as unsigned long long (64-bit, standard for unsigned long in C++ on most systems)
                # Pack payload as raw bytes
                f_out.write(struct.pack("<Q", key)) # '<Q' is for little-endian unsigned long long
                f_out.write(payload_data)
    except IOError as e:
        print(f"Error writing to file {args.output}: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error during data generation: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

    # print(f"Successfully generated {num_elements} records with payload {args.payload} bytes (seed {args.seed}, dist {args.distribution}) to {args.output}", file=sys.stderr)

if __name__ == "__main__":
    main()
