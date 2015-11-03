#!/bin/bash
mkdir -p build && cd build

# build & test library
cmake ..
make
make test

# build library + examples (examples currently require Linux)
cmake .. -DBUILD_EXAMPLES=ON
make
make test
make install
