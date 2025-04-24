#!/usr/bin/env bash
set -euo pipefail

#ensure script runs in its own directory
cd "$(dirname "$0")" || exit 1

# Build the application
make clean
echo "Building minizp..."
make app

BIN="$(pwd)/minizp"

# Prepare temporary test directory
TMPDIR=$(mktemp -d -t minizp_test.XXXX)
echo "Using temporary test dir: $TMPDIR"
trap "rm -rf $TMPDIR" EXIT

cd "$TMPDIR"

echo "Creating test files..."
# Small file (~512KB)
dd if=/dev/urandom of=small1.dat bs=512K count=1 status=none
# Large file (~100MB)
dd if=/dev/urandom of=bigfile.dat bs=2M count=50 status=none

# Nested directories for recursion tests
mkdir -p nested/emptydir nested/other
# Additional file in nested directory
dd if=/dev/urandom of=nested/other/file2.dat bs=1M count=2 status=none

# Symlink edge case
ln -s nested nested_symlink

# Helper function to run and verify commands
run_test() {
  echo "Running: $*"
  if ! $*; then
    echo "FAIL: $*"
    exit 1
  else
    echo "PASS"
  fi
}

# 1. Test compress preserve original
run_test "$BIN -C 0 small1.dat"
[ -f small1.dat.zip ] && [ -f small1.dat ] || { echo "FAIL: compress preserve"; exit 1; }

# 2. Test compress and remove original
run_test "$BIN -C 1 bigfile.dat"
[ -f bigfile.dat.zip ] && [ ! -f bigfile.dat ] || { echo "FAIL: compress remove"; exit 1; }

# 3. Test decompress preserve original
run_test "$BIN -D 0 bigfile.dat.zip"
[ -f bigfile.dat ] && [ -f bigfile.dat.zip ] || { echo "FAIL: decompress preserve"; exit 1; }

# 4. Test decompress and remove archive
run_test "$BIN -D 1 small1.dat.zip"
[ -f small1.dat ] && [ ! -f small1.dat.zip ] || { echo "FAIL: decompress remove"; exit 1; }

# 5. Test recursive compression on directory
run_test "$BIN -r 1 -C 0 nested"
[ -f nested/other/file2.dat.zip ] || { echo "FAIL: recursive compress"; exit 1; }

# 6. Test mixed input order (directory first, then file)
# Recreate bigfile.dat for testing
cp "$TMPDIR/bigfile.dat" nested/
run_test "$BIN -r 1 -C 0 nested bigfile.dat"
[ -f nested/bigfile.dat.zip ] && [ -f bigfile.dat.zip ] || { echo "FAIL: mixed order compress"; exit 1; }

echo "All tests passed successfully."
