#! /bin/bash

git clone https://github.com/endia-org/Endia.git > /dev/null 2>&1

cd Endia

git checkout nightly

cd ..

mojo package ./Endia/endia -o endia.📦

rm -rf Endia
