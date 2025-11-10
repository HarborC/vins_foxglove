#!/bin/bash

BASE_DIR=$(cd $(dirname $0);pwd)
cd ${BASE_DIR}

sudo rm -r build
mkdir build
cd build
cmake .. 
make -j2
sudo make install
cd ..
sudo rm -r build