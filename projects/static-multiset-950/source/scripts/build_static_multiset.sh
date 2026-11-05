#!/usr/bin/env bash
# Build only this task's targets; run device tests sequentially.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:-test}"
case "$mode" in build|test|perf|all) ;; *) echo 'Usage: build_static_multiset.sh [build|test|perf|all]' >&2; exit 2;; esac
: "${ASCEND_HOME_PATH:?Source CANN set_env.sh first}"
: "${Catch2_DIR:?Set Catch2_DIR to the installed Catch2 v3.5.4 CMake directory}"
build="${MULTISET_BUILD_DIR:-$root/build_multiset}"
logs="${MULTISET_LOG_DIR:-$root/output/static_multiset}"
mkdir -p "$logs"
compiler="${CXX:-$ASCEND_HOME_PATH/bin/bisheng}"
cmake -S "$root" -B "$build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$compiler" -DCatch2_DIR="$Catch2_DIR" \
  -DROARING_BITMAP_GENERATE_TESTDATA=OFF -DBUILD_TESTS=ON -DBUILD_PERFORMANCE=ON \
  -DBISHENG_AICORE_ARCH="${BISHENG_AICORE_ARCH:-dav-3510}" 2>&1 | tee "$logs/configure.log"
targets=()
for f in "$root"/tests/static_multiset/*_test.cpp; do
  name="$(basename "$f" .cpp)"
  targets+=("collection_tests_static_multiset_$name")
done
for f in "$root"/tests/performance/static_multiset/perf_*.cpp; do
  name="$(basename "$f" .cpp)"
  targets+=("static_multiset_$name")
done
cmake --build "$build" -j"${BUILD_JOBS:-2}" --target "${targets[@]}" 2>&1 | tee "$logs/build.log"
if [[ "$mode" == test || "$mode" == all ]]; then
  # Override with 0 to include runtime teardown/reinitialization for every case.
  export COLLECTION_TEST_REUSE_ACL="${COLLECTION_TEST_REUSE_ACL:-1}"
  ctest --test-dir "$build" -R '^collection_tests_static_multiset_' \
    --output-on-failure --timeout "${MULTISET_TEST_TIMEOUT:-3600}" -j1 \
    --output-junit "$logs/functional.xml" 2>&1 | tee "$logs/functional.log"
fi
if [[ "$mode" == perf || "$mode" == all ]]; then
  unset COLLECTION_TEST_REUSE_ACL
  for binary in "$build"/static_multiset/static_multiset_perf_*; do
    name="$(basename "$binary")"
    timeout -k 30s "${MULTISET_PERF_TIMEOUT:-900}" "$binary" 2>&1 | tee "$logs/$name.log"
  done
fi
