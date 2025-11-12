#!/bin/bash

BASE_DIR=$(cd $(dirname $0);pwd)
cd ${BASE_DIR}

sudo rm -rf build
rm -f ./src/proto/foxglove/*.pb.cc ./src/proto/foxglove/*.pb.h

# 生成 Ninja 构建系统
mkdir -p build && cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DProtobuf_PROTOC_EXECUTABLE=/usr/bin/protoc \
  ..

# 构建（Ninja 自动并行）
ninja

# 安装
sudo ninja install

# 回到上级并清理
cd ..
sudo rm -rf build
rm -f ./src/proto/foxglove/*.pb.cc ./src/proto/foxglove/*.pb.h