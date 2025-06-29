#!/bin/bash

BASE_DIR=$(cd $(dirname $0);pwd)
cd ${BASE_DIR}

sudo apt-get install -y libasio-dev libwebsocketpp-dev nlohmann-json3-dev
sudo apt-get install -y libeigen3-dev libopencv-dev libboost-all-dev libpcl-dev
sudo apt-get install -y autoconf automake libtool m4
sudo apt-get install -y libprotobuf-dev protobuf-compiler