#!/bin/bash

BASE_DIR=$(cd $(dirname $0);pwd)
cd ${BASE_DIR}

rm -r build
rm -r ./src/proto/foxglove/*.pb.cc
rm -r ./src/proto/foxglove/*.pb.h

mkdir build
cd build
cmake .. -D Protobuf_PROTOC_EXECUTABLE=/usr/bin/protoc
make -j2
sudo make install
cd ..

# rm -r build
# rm -r ./src/proto/foxglove/*.pb.cc
# rm -r ./src/proto/foxglove/*.pb.h