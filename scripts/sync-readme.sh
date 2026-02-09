#!/bin/bash
# Syncs code examples into README.md
# Usage: ./scripts/sync-readme.sh

set -e

README="README.md"
MARKER_START="<!-- BEGIN:examples/basic.rs -->"
MARKER_END="<!-- END:examples/basic.rs -->"
EXAMPLE="examples/basic.rs"

if [[ ! -f "$README" ]]; then
    echo "Error: $README not found"
    exit 1
fi

if [[ ! -f "$EXAMPLE" ]]; then
    echo "Error: $EXAMPLE not found"
    exit 1
fi

# Build the replacement block
REPLACEMENT="$MARKER_START
\`\`\`rust
$(cat "$EXAMPLE")
\`\`\`
$MARKER_END"

# Use awk to replace content between markers
awk -v start="$MARKER_START" -v end="$MARKER_END" -v repl="$REPLACEMENT" '
    $0 == start { found=1; print repl; next }
    $0 == end { found=0; next }
    !found { print }
' "$README" > "${README}.tmp"

mv "${README}.tmp" "$README"
echo "Synced $EXAMPLE into $README"
