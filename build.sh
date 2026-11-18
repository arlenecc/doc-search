#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
swift build -c release
echo "可执行文件: $(pwd)/.build/release/DocSearch"
