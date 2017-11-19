#! /bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")" // open current shell dir
rm -rf build
mkdir build
cd build
cmake ..
make
chmod +x Optim1GaussianSmooth
./Optim1GaussianSmooth $@
