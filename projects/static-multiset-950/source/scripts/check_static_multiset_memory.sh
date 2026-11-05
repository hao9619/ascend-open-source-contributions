#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${ASCEND_HOME_PATH:?Source CANN environment first}"
: "${Catch2_DIR:?Set Catch2_DIR}"
build="$root/build_multiset_sanitize"
logs="$root/output/static_multiset/sanitizer"
mkdir -p "$logs"
chmod 750 "$logs"
cmake -S "$root" -B "$build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$ASCEND_HOME_PATH/bin/bisheng" \
  -DCMAKE_CXX_FLAGS='--cce-enable-sanitizer -g' -DCatch2_DIR="$Catch2_DIR" \
  -DROARING_BITMAP_GENERATE_TESTDATA=OFF -DBUILD_TESTS=ON -DBUILD_PERFORMANCE=OFF \
  > "$logs/configure.log" 2>&1
cmake --build "$build" -j2 --target collection_tests_static_multiset_regression_test \
  > "$logs/build.log" 2>&1
export COLLECTION_TEST_REUSE_ACL=1
set +e
timeout -k 15s 120s mssanitizer -t memcheck --check-device-heap=yes --leak-check=yes \
  --log-file="$logs/device-memcheck.log" -- \
  "$build/tests/collection_tests_static_multiset_regression_test" 'static_multiset stable*' \
  > "$logs/device-run.log" 2>&1
result=$?
# Some mssanitizer versions return zero even when the child fails to start.
if ! grep -q 'All tests passed' "$logs/device-run.log"; then
  if [[ "$result" == 0 ]]; then result=1; fi
fi
printf 'MSSANITIZER_EXIT_CODE=%s\n' "$result" | tee "$logs/exit.txt"
cat "$logs/device-run.log"
exit "$result"
