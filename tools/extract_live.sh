#!/bin/bash
# Extract game state from a TripleA .tsvg save file to JSON.
# Usage: ./tools/extract_live.sh <input.tsvg> <output.json>

cd "$(dirname "$0")/.."

TRIPLEA_DIR="triplea"
PROJECT_JARS=$(find "$TRIPLEA_DIR" -name "*.jar" -path "*/build/libs/*" 2>/dev/null | tr '\n' ':')
CACHE_JARS=$(find ~/.gradle/caches/modules-2/files-2.1 -name "*.jar" 2>/dev/null | tr '\n' ':')
CP="tools/classes:${PROJECT_JARS}${CACHE_JARS}"

java -cp "$CP" tools.SaveToJson "$1" "$2" 2>/dev/null
