#!/bin/bash

BASE_DIR=$(cd $(dirname $0);pwd)

# cd ${BASE_DIR}/thirdparty/kalibrlib
# sh build.sh

# cd ${BASE_DIR}/thirdparty/foxglove
# sh build.sh

# cd ${BASE_DIR}/thirdparty/ethz_apriltag2
# sh build.sh

# cd ${BASE_DIR}/thirdparty/apriltag
# sh build.sh

cd ${BASE_DIR}
# rm -rf build
mkdir -p build && cd build
clear
cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="../install" .. 
ninja -j2
# sudo ninja install
# cmake  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="../install" ..
# make -j2
# sudo make install