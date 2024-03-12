#!/bin/bash
# Licensed under the Apache-2.0 license

cargo install --git https://github.com/chipsalliance/caliptra-sw --root /tmp/caliptra-file-header-fix caliptra-file-header-fix

echo Running file-header-fix
/tmp/caliptra-file-header-fix/bin/caliptra-file-header-fix "$@"
