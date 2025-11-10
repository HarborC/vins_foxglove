#!/bin/bash

BASE_DIR=$(cd $(dirname $0);pwd)
cd ${BASE_DIR}

sudo rm -r build
sudo rm -r ./src/proto/foxglove/*.pb.cc
sudo rm -r ./src/proto/foxglove/*.pb.h

mkdir build
cd build
cmake .. -D Protobuf_PROTOC_EXECUTABLE=/usr/bin/protoc
make -j2
sudo make install
cd ..

sudo rm -r build
sudo rm -r ./src/proto/foxglove/*.pb.cc
sudo rm -r ./src/proto/foxglove/*.pb.h