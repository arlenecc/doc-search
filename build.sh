#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

# Release 构建优化（体积优先）：
#   -Osize                           优化级别 = -Osize（等同 Xcode Optimization Level）
#   -gnone                           不生成调试信息（等同 Strip Debug Symbols）
#   -cross-module-optimization       跨模块优化
#   -lto=llvm-thin                   链接时优化 LTO（等同 Xcode LTO = Incremental）
#   --enable-experimental-strip-products  最终产物剥离调试符号
swift build -c release \
  -Xswiftc -Osize \
  -Xswiftc -gnone \
  -Xswiftc -cross-module-optimization \
  -Xswiftc -lto=llvm-thin \
  --enable-experimental-strip-products

echo "可执行文件: $(pwd)/.build/release/DocSearch"
ls -lh .build/release/DocSearch
