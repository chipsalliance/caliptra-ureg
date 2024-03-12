#!/bin/bash
# Licensed under the Apache-2.0 license

set -e

for p in ./ lib/*; do
  (
    cd "$p"
    cargo build
    cargo test
    cargo fmt --check
    cargo clippy -- -D warnings
  )
done

# Fix license headers
ci-tools/file-header-fix.sh --check
