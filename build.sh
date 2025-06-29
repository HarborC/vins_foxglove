#!/bin/bash

BASE_DIR=$(cd $(dirname $0);pwd)
cd ${BASE_DIR}

# sudo apt-get update -y
# sudo apt install -y libunwind-dev
# sudo apt-get install -y cmake libgoogle-glog-dev libgflags-dev libatlas-base-dev libeigen3-dev libsuitesparse-dev libceres-dev
# sudo apt-get install -y libeigen3-dev libopencv-dev libboost-all-dev libpcl-dev

# open_vins
cd ov_msckf
mkdir build
cd build
cmake ..
make -j4