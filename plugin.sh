#!/bin/sh
here="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
sys="$(uname -s)"
case "$sys" in
MINGW* | MSYS* | CYGWIN*) rel="x86_64-pc-windows-msvc/devin-sub.exe" ;;
Darwin*)
    case "$(uname -m)" in
    arm64) rel="aarch64-apple-darwin/devin-sub" ;;
    *) rel="x86_64-apple-darwin/devin-sub" ;;
    esac ;;
*)
    case "$(uname -m)" in
    aarch64 | arm64) rel="aarch64-unknown-linux-musl/devin-sub" ;;
    *) rel="x86_64-unknown-linux-musl/devin-sub" ;;
    esac ;;
esac
if [ -x "$here/$rel" ]; then exec "$here/$rel" "$@"; fi
for cand in "$here/target/release/devin-sub.exe" "$here/target/release/devin-sub" "$here/target/debug/devin-sub.exe" "$here/target/debug/devin-sub"; do
    if [ -x "$cand" ]; then exec "$cand" "$@"; fi
done
echo "devin-sub binary not found in $here" >&2
exit 1
