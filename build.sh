#!/bin/bash

BASE_DIR=$(cd $(dirname $0);pwd)

# cd ${BASE_DIR}/thirdparty/kalibrlib
# sh build.sh

cd ${BASE_DIR}
# rm -rf build
mkdir -p build && cd build
clear
cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="../install" .. 
ninja -j4
sudo ninja install