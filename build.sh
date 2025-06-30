#!/bin/bash

BASE_DIR=$(cd $(dirname $0);pwd)
cd ${BASE_DIR}

# open_vins
cd ov_msckf
mkdir build
cd build
cmake ..
make -j4