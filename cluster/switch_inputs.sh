#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   bash cluster/switch_inputs.sh <SRC_OUTPUT_DIR> <TARGET_INPUT_DIR>
# 说明：
#   将某次推理输出目录拷贝到指定输入目录，便于下游统一从 input 目录读取并快速切换不同结果。
#   若目标目录存在，将被清空后再复制。

SRC_DIR=${1:-}
DST_DIR=${2:-}

if [ -z "$SRC_DIR" ] || [ -z "$DST_DIR" ]; then
  echo "用法：bash cluster/switch_inputs.sh <SRC_OUTPUT_DIR> <TARGET_INPUT_DIR>" >&2
  exit 2
fi

if [ ! -d "$SRC_DIR" ]; then
  echo "[ERROR] 源输出目录不存在：$SRC_DIR" >&2
  exit 3
fi

mkdir -p "$DST_DIR"
# 清空目标目录
rm -rf "$DST_DIR"/*
# 复制全部内容（包括 bbox2d.json、annotated.png 等）
cp -r "$SRC_DIR"/* "$DST_DIR"/

echo "已将输出内容复制到输入目录：$DST_DIR"