#!/bin/bash

if [ ! -e "mnist_nn" ]; then 
    make clean
    make
fi

if [ "$?" -ne "0" ]; then
    echo ""
    echo "Compilation of base C implementation failed. "
    exit 1
fi

./mnist_nn $@
