#!/bin/bash

BASE_DIR=$(cd $(dirname $0);pwd)
cd ${BASE_DIR}

sudo apt install -y ninja-build

# apriltag
cmake -B build -GNinja -DCMAKE_BUILD_TYPE=Release
sudo cmake --build build --target install
sudo rm -r build
