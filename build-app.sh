#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

BIN=".build/release/DocSearch"
APP="dist/DocSearch.app"
CONTENTS="$APP/Contents"

if [[ ! -x "$BIN" ]]; then
  echo "未找到 $BIN，请先运行 ./build.sh" >&2
  exit 1
fi

echo "==> 清理旧产物"
rm -rf "$APP" dist/DocSearch.dmg

echo "==> 创建 .app 结构"
mkdir -p "$CONTENTS/MacOS" "$CONTENTS/Frameworks" "$CONTENTS/Resources"

echo "==> 拷贝可执行文件"
cp "$BIN" "$CONTENTS/MacOS/DocSearch"

echo "==> 拷贝动态库 framework"
cp -a .build/release/llama.framework "$CONTENTS/Frameworks/"

echo "==> 拷贝资源 bundle（metallib / tokenizer / 停用词等）"
for b in DocSearch_DocSearch ZIPFoundation_ZIPFoundation mlx-swift_Cmlx swift-crypto_Crypto swift-transformers_Hub; do
  cp -a ".build/release/$b.bundle" "$CONTENTS/Resources/"
done

echo "==> 写入 Info.plist"
cat > "$CONTENTS/Info.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>zh_CN</string>
    <key>CFBundleExecutable</key>
    <string>DocSearch</string>
    <key>CFBundleIdentifier</key>
    <string>local.docsearch.DocSearch</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>DocSearch</string>
    <key>CFBundleDisplayName</key>
    <string>DocSearch</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>14.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSPrincipalClass</key>
    <string>NSApplication</string>
    <key>LSUIElement</key>
    <false/>
</dict>
</plist>
EOF

echo "==> 修正 rpath：加载 Frameworks 下的 llama"
install_name_tool -add_rpath "@executable_path/../Frameworks" "$CONTENTS/MacOS/DocSearch"

echo "==> ad-hoc 代码签名"
codesign --force --deep --sign - "$APP"

echo "==> 生成 dmg"
hdiutil create -volname DocSearch -srcfolder "$APP" -ov -format UDZO dist/DocSearch.dmg

echo "==> 完成"
echo "app: $(pwd)/$APP"
echo "dmg: $(pwd)/dist/DocSearch.dmg"
