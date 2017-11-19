#! /bin/bash
set -e  // if return code is not 0 then exit shell
set -x //  display all parameters of the cmd after exec the current cmd
cd "$(dirname "${BASH_SOURCE[0]}")" // open current shell dir
rm -rf build
mkdir build
cd build
cmake ..
make
chmod +x NormalSquareSum
./NormalSquareSum
