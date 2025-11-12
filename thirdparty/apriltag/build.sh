#!/bin/bash

BASE_DIR=$(cd $(dirname $0);pwd)
cd ${BASE_DIR}

sudo rm -rf build
mkdir build
cd build

# 生成 Ninja 构建系统
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..

# 构建（自动并行，不用手动写 -j）
ninja

# 安装
sudo ninja install

# 返回并清理构建目录
cd ..
sudo rm -rf build